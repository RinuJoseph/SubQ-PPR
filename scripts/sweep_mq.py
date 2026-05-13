#!/usr/bin/env python
"""scripts/sweep_mq.py — sweep mq_top over a range, report recall.

Runs the full retrieval pipeline (pool → graph → LLM fact filter → PPR) for
each mq_top value over all queries and prints a comparison table.

Usage:
  python scripts/sweep_mq.py --config configs/default.yaml \
      --mq 5 10 20 50 100 200
  python scripts/sweep_mq.py --config configs/default.yaml --limit 200  # quick scan
"""
import argparse
import copy
import sys
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from sqppr.utils import load_config, save_json
from sqppr.data import load_dataset_with_decomp
from sqppr.embedding import passage_embeddings_and_keys
from sqppr.retrieval import retrieve_one
from sqppr.metrics import recall_at_k

KS = [5, 10, 20, 30, 50]


def run_one_mq(mq_top, queries, corpus, passage_embs, passage_keys, cfg, workers):
    cfg_local = copy.deepcopy(cfg)
    cfg_local["pool"]["mq_top"] = mq_top
    rows = []
    t0 = time.time(); n_err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(retrieve_one, q, corpus, passage_embs, passage_keys, cfg_local): q
                  for q in queries}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception:
                n_err += 1
    # Aggregate
    out = {"mq_top": mq_top, "n": len(rows), "errors": n_err,
            "time_s": round(time.time() - t0, 1)}
    for k in KS:
        vs = []
        for r in rows:
            gold = [g["chunk_id"] for g in r["gold"] if g.get("chunk_id")]
            v = recall_at_k(r["ranking"], gold, k)
            if v is not None: vs.append(v)
        out[f"recall@{k}"] = round(sum(vs)/len(vs), 6) if vs else None
    cov = [r["final_pool_coverage"] for r in rows if r["final_pool_coverage"] is not None]
    out["pool_coverage"] = round(sum(cov)/len(cov), 6) if cov else None
    out["avg_pool_size"] = round(sum(r["n_pool"] for r in rows)/max(1, len(rows)), 2)
    out["avg_kept_facts"] = round(sum(r["n_facts_kept_by_llm"] for r in rows)/max(1, len(rows)), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs/default.yaml"))
    ap.add_argument("--mq", type=int, nargs="+",
                    default=[5, 10, 20, 50, 100, 200],
                    help="mq_top values to sweep")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of queries (for a fast scan)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dataset", default=None,
                    help="Dataset name (overrides config.dataset.name).")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_override=args.dataset)
    if args.workers: cfg["workers"]["llm_filter"] = args.workers
    workers = cfg["workers"]["llm_filter"]

    print(f"[load] config: {cfg['_config_path']}")
    corpus, queries = load_dataset_with_decomp(
        cfg["dataset"]["name"], Path(cfg["dataset"]["root"]),
        Path(cfg["cache"]["decompositions"]),
    )
    if args.limit: queries = queries[:args.limit]
    print(f"[load] {len(corpus)} passages   {len(queries)} queries   workers={workers}")
    print("[load] passage embeddings …")
    passage_embs, passage_keys = passage_embeddings_and_keys(cache_dir=Path(cfg["cache"]["passage_embeddings"]), corpus=corpus)

    print(f"\n[sweep] mq_top ∈ {args.mq}")
    results = []
    for mq in args.mq:
        print(f"\n[run] mq={mq} …", flush=True)
        r = run_one_mq(mq, queries, corpus, passage_embs, passage_keys, cfg, workers)
        results.append(r)
        print(f"  mq={mq:>4d}  pool={r['avg_pool_size']:>6.2f}  "
                f"R@5={r['recall@5']:.4f}  R@10={r['recall@10']:.4f}  "
                f"R@20={r['recall@20']:.4f}  R@30={r['recall@30']:.4f}  "
                f"R@50={r['recall@50']:.4f}  cov={r['pool_coverage']:.4f}  "
                f"({r['time_s']}s, errs={r['errors']})", flush=True)

    # Table
    print()
    print("=" * 100)
    print(f"  mq_top sweep   n_queries={len(queries)}")
    print("=" * 100)
    header = f"  {'mq':>4s}  {'pool':>7s}  {'kept':>5s}  " + "  ".join(f"R@{k:<3d}" for k in KS) + "  cov     time"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        cells = "  ".join(f"{r[f'recall@{k}']:.4f}" if r[f'recall@{k}'] is not None else "  n/a" for k in KS)
        print(f"  {r['mq_top']:>4d}  {r['avg_pool_size']:>7.2f}  {r['avg_kept_facts']:>5.2f}  "
                f"{cells}  {r['pool_coverage']:.4f}  {r['time_s']:>5.1f}s")
    print()
    print(f"  HippoRAG-2 reported: R@5 = 0.7418   R@10 = 0.8305   R@20 = 0.8809")

    out = Path(cfg["output_dir"]) / "sweep_mq.json"
    save_json({"n_queries": len(queries), "workers": workers, "results": results}, out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
