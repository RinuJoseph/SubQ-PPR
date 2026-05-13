#!/usr/bin/env python
"""scripts/run_qa.py — final QA on retrieved top-5 (no retrieval).

Reads results/retrieval/summary.json, takes each query's top_5 chunk_ids,
runs gpt-4o-mini with the rag_qa_musique prompt, computes F1/EM.

Usage:
  python scripts/run_qa.py --config configs/default.yaml
  python scripts/run_qa.py --config configs/default.yaml --limit 50
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from sqppr.utils import load_config, save_json
from sqppr.data import load_dataset_with_decomp
from sqppr.qa import qa_one
from sqppr.metrics import compute_f1, compute_em


def hop_class(qid):
    m = re.match(r"^(\d+)hop", qid)
    return f"{m.group(1)}hop" if m else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None,
                    help="Dataset name (overrides config.dataset.name). "
                         "Supported: musique, hotpotqa, 2wikimultihopqa, hover, popqa, nq_rear")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_override=args.dataset)
    if args.workers: cfg["workers"]["qa"] = args.workers

    print(f"[load] config: {cfg['_config_path']}  dataset={cfg['dataset']['name']}")
    corpus, qlist = load_dataset_with_decomp(
        cfg["dataset"]["name"], Path(cfg["dataset"]["root"]),
        Path(cfg["cache"]["decompositions"]),
    )
    queries = {q["qid"]: q for q in qlist}

    retrieval_path = Path(cfg["output_dir"]) / "retrieval" / "summary.json"
    if not retrieval_path.exists():
        sys.exit(f"missing retrieval results: {retrieval_path}\n"
                 f"run scripts/run_retrieval.py first.")
    retrieval = json.load(open(retrieval_path))
    rows = retrieval["per_query"]
    if args.limit: rows = rows[:args.limit]

    print(f"[load] {len(rows)} queries with saved top_5")
    print(f"[run] QA model = {cfg['llm']['qa_model']}, top-{cfg['qa']['top_k_passages']} "
            f"passages, {cfg['workers']['qa']} workers")

    out_dir = Path(cfg["output_dir"]) / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)

    def task(row):
        qid = row["qid"]
        qr = queries.get(qid)
        if qr is None:
            return qid, None, "qid not in queries"
        # MuSiQue stores per-hop answers in qr['gold'][i]['answer']; the
        # final answer is the last hop's. Every other dataset (hotpotqa,
        # 2wiki, hover, popqa, nq_rear, lveval, narrativeqa) puts the gold
        # answer on the query itself in qr['answer']. Use the per-hop list
        # if present, else fall back to qr['answer'].
        gold_answers = [g["answer"] for g in qr["gold"] if g.get("answer")]
        if gold_answers:
            final_gold = gold_answers[-1]
        else:
            final_gold = qr.get("answer") or ""
            if isinstance(final_gold, list) and final_gold:
                final_gold = final_gold[0]
            elif not isinstance(final_gold, str):
                final_gold = str(final_gold) if final_gold is not None else ""
        top = row["top_5"][:cfg["qa"]["top_k_passages"]]
        try:
            r = qa_one(
                qr["query"], top, corpus,
                model=cfg["llm"]["qa_model"],
                temperature=cfg["qa"]["temperature"],
                max_completion_tokens=cfg["qa"]["max_completion_tokens"],
            )
        except Exception as e:
            return qid, None, str(e)
        f1 = compute_f1(r["predicted_answer"], final_gold)
        em = compute_em(r["predicted_answer"], final_gold)
        return qid, {
            "qid": qid, "question": qr["query"],
            "gold_answer": final_gold,
            "all_hop_answers": gold_answers,
            "predicted_answer": r["predicted_answer"],
            "raw_response": r["raw_response"],
            "f1": f1, "em": em,
            "top_5_titles": [p["title"] for p in top],
            "top_5_gold_flags": [p["is_gold"] for p in top],
        }, None

    results = {}
    t0 = time.time(); n_done = n_err = 0
    with ThreadPoolExecutor(max_workers=cfg["workers"]["qa"]) as ex:
        futs = {ex.submit(task, r): r for r in rows}
        for fut in as_completed(futs):
            qid, rec, err = fut.result()
            n_done += 1
            if err:
                n_err += 1
                if n_err < 5: print(f"  [err] {qid}: {err}", flush=True)
                continue
            results[qid] = rec
            if n_done % 50 == 0 or n_done == len(rows):
                mf = sum(r["f1"] for r in results.values()) / max(1, len(results))
                me = sum(r["em"] for r in results.values()) / max(1, len(results))
                print(f"  [{n_done:4d}/{len(rows)}] {time.time()-t0:.0f}s  "
                        f"F1={mf:.4f}  EM={me:.4f}  errs={n_err}", flush=True)

    n = len(results)
    mean_f1 = sum(r["f1"] for r in results.values()) / max(1, n)
    mean_em = sum(r["em"] for r in results.values()) / max(1, n)

    by_hop = defaultdict(list)
    for qid, r in results.items():
        by_hop[hop_class(qid)].append((r["f1"], r["em"]))

    print()
    print("=" * 80)
    print(f"  QA  n={n}  errors={n_err}  model={cfg['llm']['qa_model']}")
    print("=" * 80)
    print(f"  Mean F1: {mean_f1:.4f}")
    print(f"  Mean EM: {mean_em:.4f}")
    print(f"\n  Per-hop:")
    for hc in ("2hop", "3hop", "4hop"):
        vs = by_hop.get(hc, [])
        if not vs: continue
        f1 = sum(x[0] for x in vs) / len(vs)
        em = sum(x[1] for x in vs) / len(vs)
        print(f"    {hc:<8s}  n={len(vs):>4d}  F1={f1:.4f}  EM={em:.4f}")
    print(f"\n  HippoRAG-2 reported (full corpus + Llama-3.3-70B filter):")
    print(f"    F1 = 0.4809   EM = 0.343")

    save_json({
        "qa_model":   cfg["llm"]["qa_model"],
        "top_k":      cfg["qa"]["top_k_passages"],
        "n_queries":  n,
        "mean_f1":    round(mean_f1, 6),
        "mean_em":    round(mean_em, 6),
        "per_query":  [results[k] for k in sorted(results.keys())],
    }, out_dir / "qa_results.json")

    with open(out_dir / "qa_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "qid", "question", "gold_answer", "predicted_answer", "f1", "em"])
        w.writeheader()
        for qid in sorted(results.keys()):
            r = results[qid]
            w.writerow({k: r[k] for k in ["qid", "question", "gold_answer",
                                            "predicted_answer", "f1", "em"]})
    print(f"\nsaved -> {out_dir/'qa_results.json'}")
    print(f"saved -> {out_dir/'qa_results.csv'}")


if __name__ == "__main__":
    main()
