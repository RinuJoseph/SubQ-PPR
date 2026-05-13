#!/usr/bin/env python
"""scripts/run_retrieval.py — batch retrieval over all queries.

Reads from caches (NER, triples, embeddings) populated by build_cache.py.
Calls gpt-4o-mini for the LLM fact filter (one call per query).
Writes per-query results + aggregate to results/retrieval/.

Usage:
  python scripts/run_retrieval.py --config configs/default.yaml
  python scripts/run_retrieval.py --config configs/default.yaml --limit 50
  python scripts/run_retrieval.py --config configs/default.yaml --workers 32
"""
import argparse
import csv
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
from sqppr.embedding import passage_embeddings_and_keys, _get_backend
from sqppr.retrieval import retrieve_one
from sqppr.metrics import recall_at_k
from sqppr.cache_prep import prepare_cache

KS = [5, 10, 15, 20, 30, 40, 50]


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
    ap.add_argument("--workers", type=int, default=None,
                    help="Override LLM filter concurrency from config")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_override=args.dataset)
    if args.workers: cfg["workers"]["llm_filter"] = args.workers

    print(f"[load] config: {cfg['_config_path']}")
    corpus, queries = load_dataset_with_decomp(
        cfg["dataset"]["name"], Path(cfg["dataset"]["root"]),
        Path(cfg["cache"]["decompositions"]),
    )
    if args.limit: queries = queries[:args.limit]
    print(f"[load] {len(corpus)} passages   {len(queries)} queries")

    # Idempotent cache prep — builds anything missing for these queries:
    #   decompositions, passage embs, text embs, ner_doc, triples
    passage_embs, passage_keys, queries = prepare_cache(
        queries, corpus, cfg, workers=cfg["workers"]["llm_filter"],
    )

    # Pre-warm the NV-Embed-v2 model on the main thread BEFORE spawning
    # workers — `retrieve_one` calls `embed_texts` for fact embeddings
    # (kind="query_to_fact"), and without warm-up 16 workers race on cold
    # model load, thrashing GPU memory. Lock in sqppr.embedding._get_backend
    # also defends, but this is the cheaper path.
    print("[warm] loading NV-Embed-v2 on main thread …", flush=True)
    _get_backend()

    out_dir = Path(cfg["output_dir"]) / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[run] retrieval — mq={cfg['pool']['mq_top']}, sq={cfg['pool']['sq_top']}, "
            f"damping={cfg['ppr']['damping']}, pw={cfg['ppr']['passage_node_weight']}, "
            f"scoring={cfg['ppr']['scoring_rule']}")

    def task(qr):
        try:
            return retrieve_one(qr, corpus, passage_embs, passage_keys, cfg), None
        except Exception as e:
            return {"qid": qr["qid"], "error": str(e)}, str(e)

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg["workers"]["llm_filter"]) as ex:
        futs = {ex.submit(task, q): q for q in queries}
        n_done = n_err = 0
        for fut in as_completed(futs):
            rec, err = fut.result()
            n_done += 1
            if err:
                n_err += 1
                if n_err < 5: print(f"  [err] {rec.get('qid')}: {err}", flush=True)
                continue
            rows.append(rec)
            if n_done % 50 == 0 or n_done == len(queries):
                r5 = sum(recall_at_k(r["ranking"], [g["chunk_id"] for g in r["gold"] if g.get("chunk_id")], 5) or 0.0
                          for r in rows) / max(1, len(rows))
                print(f"  [{n_done:4d}/{len(queries)}] {time.time()-t0:.0f}s  R@5={r5:.4f}", flush=True)
    print(f"[run] done in {time.time()-t0:.0f}s  errors={n_err}")

    # Aggregate recall
    def agg(k):
        vs = []
        for r in rows:
            gold = [g["chunk_id"] for g in r["gold"] if g.get("chunk_id")]
            v = recall_at_k(r["ranking"], gold, k)
            if v is not None: vs.append(v)
        return round(sum(vs)/len(vs), 6) if vs else None
    aggregate = {f"recall@{k}": agg(k) for k in KS}

    cov_vals = [r["final_pool_coverage"] for r in rows if r["final_pool_coverage"] is not None]
    avg_cov = round(sum(cov_vals)/len(cov_vals), 6) if cov_vals else None
    avg_pool = round(sum(r["n_pool"] for r in rows)/len(rows), 2)

    # Hop-level R@5
    by_hop = defaultdict(list)
    for r in rows:
        gold = [g["chunk_id"] for g in r["gold"] if g.get("chunk_id")]
        v = recall_at_k(r["ranking"], gold, 5)
        if v is not None:
            by_hop[hop_class(r["qid"])].append(v)

    print()
    print("=" * 100)
    print(f"  RETRIEVAL  n={len(rows)}  avg_pool={avg_pool}  avg_coverage={avg_cov:.4f}")
    print("=" * 100)
    print(f"  {'metric':<14s}  " + "  ".join(f"@{k:>3d}" for k in KS))
    cells = "  ".join(f"{aggregate[f'recall@{k}']:.4f}" for k in KS)
    print(f"  {'PPR ranking':<14s}  {cells}")
    print(f"\n  hop-level R@5:")
    for hc in ("2hop", "3hop", "4hop"):
        if by_hop[hc]:
            print(f"    {hc:<8s}  n={len(by_hop[hc]):>4d}  R@5={sum(by_hop[hc])/len(by_hop[hc]):.4f}")
    print(f"\n  HippoRAG-2 reported: R@5 = 0.7418")

    # Save summary.json — strip heavy fields per row
    summary_rows = []
    for r in rows:
        summary_rows.append({
            "qid": r["qid"], "query": r["query"],
            "n_sub_questions": r["n_sub_questions"],
            "n_pool": r["n_pool"], "n_entity_nodes": r["n_entity_nodes"],
            "n_edges": r["n_edges"], "n_triples": r["n_triples"],
            "n_gold": r["n_gold"],
            "n_facts_kept_by_llm": r["n_facts_kept_by_llm"],
            "final_pool_coverage": r["final_pool_coverage"],
            "recall": {f"recall@{k}": recall_at_k(
                r["ranking"],
                [g["chunk_id"] for g in r["gold"] if g.get("chunk_id")], k)
                for k in KS},
            "top_5": r["top_5"],
            "gold_landings": r["gold_landings"],
        })
    save_json({
        "config":               cfg,
        "n_queries":            len(rows),
        "avg_pool_size":        avg_pool,
        "avg_final_pool_coverage": avg_cov,
        "aggregate":            aggregate,
        "per_query":            summary_rows,
    }, out_dir / "summary.json")

    out_csv = out_dir / "summary.csv"
    with open(out_csv, "w", newline="") as f:
        fns = ["qid", "query", "n_pool", "n_entity_nodes", "n_edges",
                "n_gold", "n_facts_kept_by_llm", "final_pool_coverage"]
        fns += [f"recall@{k}" for k in KS]
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        for r in summary_rows:
            row = {k: r.get(k) for k in fns if k in r}
            row.update({k: r["recall"][k] for k in r["recall"]})
            w.writerow(row)
    print(f"\nsaved -> {out_dir/'summary.json'}")
    print(f"saved -> {out_dir/'summary.csv'}")


if __name__ == "__main__":
    main()
