#!/usr/bin/env python
"""scripts/sweep_dpr_recall.py — pure DPR recall@K sweep.

For each query, embed the main question with NV-Embed (query_to_passage instr),
take cosine vs all 11.6K passage embeddings, compute recall@K for many Ks.
NO graph, NO PPR, NO LLM. Just dense retrieval.

Usage:
  python scripts/sweep_dpr_recall.py --config configs/default.yaml \
      --ks 5 10 20 50 100 150 200 250 300 500
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from sqppr.utils import load_config, save_json
from sqppr.data import load_dataset_with_decomp
from sqppr.embedding import passage_embeddings_and_keys, embed_text
from sqppr.metrics import recall_at_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs/default.yaml"))
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[5, 10, 20, 50, 100, 150, 200, 250, 300, 500])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dataset", default=None,
                    help="Dataset name (overrides config.dataset.name).")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_override=args.dataset)
    print(f"[load] config: {cfg['_config_path']}")
    corpus, queries = load_dataset_with_decomp(
        cfg["dataset"]["name"], Path(cfg["dataset"]["root"]),
        Path(cfg["cache"]["decompositions"]),
    )
    if args.limit: queries = queries[:args.limit]
    print(f"[load] {len(corpus)} passages   {len(queries)} queries")

    print("[load] passage embeddings …")
    passage_embs, passage_keys = passage_embeddings_and_keys(cache_dir=Path(cfg["cache"]["passage_embeddings"]), corpus=corpus)
    print(f"  embs={passage_embs.shape}")

    emb_cache_dir = Path(cfg["cache"]["embeddings_text"])
    Ks = sorted(set(args.ks))
    Kmax = max(Ks)

    print(f"\n[run] cosine top-{Kmax} for each query …")
    t0 = time.time()
    by_hop = {"2hop": [], "3hop": [], "4hop": []}
    per_query = []
    sums = {k: 0.0 for k in Ks}
    counts = {k: 0 for k in Ks}
    for i, q in enumerate(queries, 1):
        gold = [g["chunk_id"] for g in q["gold"] if g.get("chunk_id")]
        if not gold:
            continue
        emb = embed_text(q["query"], emb_cache_dir, "query_to_passage")
        scores = np.dot(passage_embs, emb.T).squeeze()
        order = np.argsort(scores)[::-1][:Kmax]
        ranking = [passage_keys[idx] for idx in order]
        row = {"qid": q["qid"]}
        for k in Ks:
            v = recall_at_k(ranking, gold, k)
            if v is not None:
                sums[k] += v; counts[k] += 1
                row[f"recall@{k}"] = v
        per_query.append(row)
        # hop class
        import re
        m = re.match(r"^(\d+)hop", q["qid"])
        hc = f"{m.group(1)}hop" if m else "unknown"
        if hc in by_hop:
            by_hop[hc].append({k: row.get(f"recall@{k}", None) for k in Ks})
        if i % 200 == 0 or i == len(queries):
            print(f"  [{i:4d}/{len(queries)}] {time.time()-t0:.0f}s", flush=True)

    print()
    print("=" * 90)
    print(f"  Pure DPR (NV-Embed-v2, query_to_passage)   n={counts[Ks[0]]}")
    print("=" * 90)
    print(f"  {'K':>5s}  {'recall':>8s}")
    aggregate = {}
    for k in Ks:
        r = sums[k] / counts[k] if counts[k] else None
        aggregate[f"recall@{k}"] = round(r, 6) if r is not None else None
        print(f"  {k:>5d}  {r:.4f}" if r is not None else f"  {k:>5d}    n/a")

    print(f"\n  Per-hop:")
    for hc, vs in by_hop.items():
        if not vs: continue
        print(f"    {hc:<8s}  n={len(vs):>4d}")
        for k in Ks:
            xs = [r[k] for r in vs if r.get(k) is not None]
            if xs:
                print(f"      R@{k:<4d}  {sum(xs)/len(xs):.4f}")

    out = Path(cfg["output_dir"]) / "sweep_dpr_recall.json"
    save_json({
        "n_queries":  counts[Ks[0]],
        "Ks":         Ks,
        "aggregate":  aggregate,
        "per_hop":    {hc: {f"recall@{k}": (sum(r[k] for r in vs if r.get(k) is not None) /
                                              max(1, sum(1 for r in vs if r.get(k) is not None)))
                              for k in Ks}
                        for hc, vs in by_hop.items() if vs},
    }, out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
