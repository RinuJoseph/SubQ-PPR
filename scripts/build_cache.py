#!/usr/bin/env python
"""scripts/build_cache.py — populate all caches needed for retrieval.

Build steps (idempotent — skips work that's already cached):
  0. Decompositions     (gpt-4o-mini, per qid)
  1. Passage embeddings (NV-Embed-v2 over all corpus chunks; bootstrap from
     HippoRAG-2 parquet when available, else GPU-encode)
  2. Per-text embeddings (queries + sub-questions, query_to_passage)
  3. NER per chunk      (gpt-4o-mini, only for chunks in any retrieval pool)
  4. Triples per chunk  (gpt-4o-mini, NER-conditioned OpenIE)

The graph build (sqppr/graph.py) uses entity nodes from triple subjects/objects
only — NER on the query/sub-q texts is intentionally NOT built here.

Usage:
  python scripts/build_cache.py --config configs/default.yaml --dataset hover
"""
import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# add repo root so `import src.*` works
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import numpy as np

from sqppr.utils import load_config
from sqppr.data import load_dataset_with_decomp
from sqppr.embedding import passage_embeddings_and_keys, embed_text, min_max_normalize, embed_passages
from sqppr.openie import _get_openie_backend, ner_chunk, triples_chunk
from sqppr.decomposition import build_all as build_decompositions


def collect_targets(queries, passage_embs, passage_keys, cfg):
    """Build the union of chunk_ids that will appear in any pool.

    The graph uses entities from OpenIE triples on these chunks only — NER
    on the query / sub-q texts is NOT consumed by the graph (see
    sqppr/graph.py), so we don't bother building it.
    """
    mq_top = cfg["pool"]["mq_top"]
    sq_top = cfg["pool"]["sq_top"]
    emb_cache_dir = Path(cfg["cache"]["embeddings_text"])
    chunk_ids = set()
    for q in queries:
        mq_emb = embed_text(q["query"], emb_cache_dir, "query_to_passage")
        mq_scores = min_max_normalize(np.dot(passage_embs, mq_emb.T).squeeze())
        for idx in np.argsort(mq_scores)[::-1][:mq_top]:
            chunk_ids.add(passage_keys[idx])
        for sub in q.get("llm_decomposition", []) or []:
            sq_emb = embed_text(sub, emb_cache_dir, "query_to_passage")
            sq_scores = min_max_normalize(np.dot(passage_embs, sq_emb.T).squeeze())
            for idx in np.argsort(sq_scores)[::-1][:sq_top]:
                chunk_ids.add(passage_keys[idx])
    return chunk_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", default=None,
                    help="Dataset name (overrides config.dataset.name). "
                         "Supported: musique, hotpotqa, 2wikimultihopqa, hover, popqa, nq_rear")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config, dataset_override=args.dataset)
    print(f"[load] config: {cfg['_config_path']}")

    corpus, queries = load_dataset_with_decomp(
        cfg["dataset"]["name"], Path(cfg["dataset"]["root"]),
        Path(cfg["cache"]["decompositions"]),
    )
    print(f"[load] dataset={cfg['dataset']['name']}  "
            f"{len(corpus)} passages   {len(queries)} queries")

    # Step 0: decompositions (idempotent; same prompt as Claude run, default
    #   model gpt-4o-mini per configs/default.yaml).
    print(f"\n[step 0] LLM decompositions  "
            f"(model={cfg['llm']['decomposition_model']}) …")
    build_decompositions(
        queries,
        model=cfg["llm"]["decomposition_model"],
        cache_dir=Path(cfg["cache"]["decompositions"]),
        workers=cfg["workers"]["decomposition"],
    )
    # Re-load queries so each record now carries its decomposition.
    _corpus, queries = load_dataset_with_decomp(
        cfg["dataset"]["name"], Path(cfg["dataset"]["root"]),
        Path(cfg["cache"]["decompositions"]),
    )
    n_with = sum(1 for q in queries if q["llm_decomposition"])
    print(f"  {n_with}/{len(queries)} queries now have a cached decomposition")

    # Step 1: passage embeddings (build if missing)
    print(f"\n[step 1] passage embeddings …")
    passage_embs, passage_keys = passage_embeddings_and_keys(
        cache_dir=Path(cfg["cache"]["passage_embeddings"]),
        corpus=corpus,
    )
    print(f"  embs shape: {passage_embs.shape}")

    print(f"\n[step 2] collect chunks touched by any pool "
            f"(mq={cfg['pool']['mq_top']}, sq={cfg['pool']['sq_top']}) …")
    chunk_ids = collect_targets(queries, passage_embs, passage_keys, cfg)
    print(f"  unique chunks needed: {len(chunk_ids)}")

    ner_doc_dir = Path(cfg["cache"]["ner_doc"])
    triples_dir = Path(cfg["cache"]["triples"])

    miss_ner_doc = [c for c in chunk_ids if not (ner_doc_dir / f"{c}.json").exists()]
    miss_triples = [c for c in chunk_ids if not (triples_dir / f"{c}.json").exists()]

    print(f"\n[cache status]")
    print(f"  ner_doc:  {len(chunk_ids) - len(miss_ner_doc)} cached / {len(miss_ner_doc)} missing")
    print(f"  triples:  {len(chunk_ids) - len(miss_triples)} cached / {len(miss_triples)} missing")

    if not (miss_ner_doc or miss_triples):
        print("\n[done] nothing to do."); return

    oie = _get_openie_backend()

    def fill_ner(c):
        try:
            ner_chunk(oie, c, corpus[c]["content"], ner_doc_dir)
            return c, None
        except Exception as e:
            return c, str(e)

    def fill_triples(c):
        try:
            ents = []
            ner_p = ner_doc_dir / f"{c}.json"
            if ner_p.exists():
                import json
                ents = json.load(open(ner_p)).get("unique_entities", [])
            triples_chunk(oie, c, corpus[c]["content"], ents, triples_dir)
            return c, None
        except Exception as e:
            return c, str(e)

    def run_pool(label, items, fn):
        if not items:
            print(f"\n[{label}] nothing to do."); return
        print(f"\n[{label}] computing {len(items)} entries with {args.workers} workers …")
        t0 = time.time(); n_done = n_err = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fn, c): c for c in items}
            for f in as_completed(futs):
                _, err = f.result()
                if err: n_err += 1
                n_done += 1
                if n_done % 200 == 0:
                    print(f"  [{n_done:4d}/{len(items)}] {time.time()-t0:.0f}s errs={n_err}",
                            flush=True)
        print(f"[{label}] done in {time.time()-t0:.0f}s.  errs={n_err}")

    run_pool("ner_doc", miss_ner_doc, fill_ner)
    run_pool("triples", miss_triples, fill_triples)

    print(f"\n[final cache state]")
    print(f"  ner_doc files:  {len(list(ner_doc_dir.glob('*.json')))}")
    print(f"  triples files:  {len(list(triples_dir.glob('*.json')))}")


if __name__ == "__main__":
    main()
