"""Idempotent cache preparation for a list of queries.

Ensures every artifact `retrieve_one` will need is present on disk for the
given queries. Re-running is cheap: existing files short-circuit immediately,
only misses trigger an LLM / encoder call.

Stages (in order):
  1. decompositions — LLM, ~1 call per missing qid
  2. passage embeddings — global, built once via embed_passages or the
     HippoRAG-2 parquet bootstrap
  3. text embeddings — NV-Embed, lazy through embed_text/embed_texts
  4. ner_doc + triples — gpt-4o-mini, only for chunks the pool will touch

NOTE: NER on the query / sub-q texts is NOT built — the graph uses entity
nodes from OpenIE triples on pool chunks only (see sqppr/graph.py).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import load_queries
from .decomposition import build_all as build_decompositions
from .embedding import (
    embed_text, embed_texts, min_max_normalize, passage_embeddings_and_keys,
)
from .openie import _get_openie_backend, ner_chunk, triples_chunk


def _collect_pool_targets(queries: List[dict], passage_embs: np.ndarray,
                            passage_keys: List[str], cfg: dict
                            ) -> set:
    """Walk each query+sub-q, take its top-K pool against passage_embs, return
    the union of chunk_ids touched. (Text embeddings are built on demand by
    embed_text below.)"""
    mq_top = cfg["pool"]["mq_top"]
    sq_top = cfg["pool"]["sq_top"]
    emb_cache_dir = Path(cfg["cache"]["embeddings_text"])
    chunk_ids: set[str] = set()
    for q in queries:
        mq_emb = embed_text(q["query"], emb_cache_dir, "query_to_passage")
        scores = min_max_normalize(np.dot(passage_embs, mq_emb.T).squeeze())
        for idx in np.argsort(scores)[::-1][:mq_top]:
            chunk_ids.add(passage_keys[idx])
        for sub in q.get("llm_decomposition", []) or []:
            sq_emb = embed_text(sub, emb_cache_dir, "query_to_passage")
            scores = min_max_normalize(np.dot(passage_embs, sq_emb.T).squeeze())
            for idx in np.argsort(scores)[::-1][:sq_top]:
                chunk_ids.add(passage_keys[idx])
    return chunk_ids


def prepare_cache(queries: List[dict], corpus: Dict[str, dict], cfg: dict,
                    workers: int = 16, verbose: bool = True) -> tuple:
    """Build (idempotently) every cache `retrieve_one` will need for these
    queries. Returns (passage_embs, passage_keys, queries_with_decomp).
    """
    def log(msg):
        if verbose: print(msg, flush=True)

    # 1. Decompositions (LLM; cached per qid) — keep only the original
    #    query subset so a small `queries` arg doesn't fan out to the full set.
    decomp_dir = Path(cfg["cache"]["decompositions"])
    missing = [q for q in queries if not q["llm_decomposition"]]
    if missing:
        log(f"[cache] {len(missing)} queries missing decompositions "
              f"-> {cfg['llm']['decomposition_model']}")
        build_decompositions(
            missing, model=cfg["llm"]["decomposition_model"],
            cache_dir=decomp_dir,
            workers=cfg["workers"].get("decomposition", workers),
        )
        qid_set = {q["qid"] for q in queries}
        all_queries = load_queries(
            Path(cfg["data"]["queries_json"]), decomp_dir, corpus,
        )
        queries = [q for q in all_queries if q["qid"] in qid_set]

    # 2. Passage embeddings (build if missing, fast bootstrap from parquet)
    passage_embs, passage_keys = passage_embeddings_and_keys(
        cache_dir=Path(cfg["cache"]["passage_embeddings"]), corpus=corpus,
    )

    # 3 + 4. Pool-driven NER + triple caches for every chunk touched by some
    #         query's pool. Graph construction uses ONLY these chunk triples
    #         — NER on the query / sub-question texts is NOT consumed by the
    #         graph (see sqppr/graph.py), so we don't build it here.
    log(f"[cache] scanning pools (mq={cfg['pool']['mq_top']}, "
          f"sq={cfg['pool']['sq_top']}) for chunk dependencies …")
    chunk_ids = _collect_pool_targets(queries, passage_embs, passage_keys, cfg)

    ner_doc_dir = Path(cfg["cache"]["ner_doc"])
    triples_dir = Path(cfg["cache"]["triples"])

    miss_ner_doc = [c for c in chunk_ids if not (ner_doc_dir / f"{c}.json").exists()]
    miss_triples = [c for c in chunk_ids if not (triples_dir / f"{c}.json").exists()]

    log(f"[cache] chunks touched: {len(chunk_ids)}  "
          f"ner_doc miss: {len(miss_ner_doc)}  "
          f"triples miss: {len(miss_triples)}")

    if not (miss_ner_doc or miss_triples):
        return passage_embs, passage_keys, queries

    oie = _get_openie_backend()

    def fill_ner(c):
        try: ner_chunk(oie, c, corpus[c]["content"], ner_doc_dir); return c, None
        except Exception as e: return c, str(e)
    def fill_triples(c):
        try:
            ents = []
            p = ner_doc_dir / f"{c}.json"
            if p.exists():
                ents = json.load(open(p)).get("unique_entities", [])
            triples_chunk(oie, c, corpus[c]["content"], ents, triples_dir)
            return c, None
        except Exception as e: return c, str(e)

    def run_pool(label, items, fn):
        if not items: return
        log(f"[cache:{label}] {len(items)} entries with {workers} workers …")
        n_err = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, x): x for x in items}
            for fut in as_completed(futs):
                _, err = fut.result()
                if err: n_err += 1
        log(f"[cache:{label}] done. errs={n_err}")

    run_pool("ner_doc", miss_ner_doc, fill_ner)
    run_pool("triples", miss_triples, fill_triples)

    # 5. Pre-encode all unique fact strings ("subj pred obj") across the union
    #    of pool chunks, with the `query_to_fact` instruction. Retrieve workers
    #    then just read the cache — no GPU contention during retrieval.
    _prepare_fact_embeddings(chunk_ids, triples_dir, cfg, verbose=verbose)

    return passage_embs, passage_keys, queries


def _prepare_fact_embeddings(chunk_ids, triples_dir, cfg, verbose=True):
    """Enumerate fact strings from every chunk's cached triples, dedupe, then
    batch-encode the misses into the per-text embeddings cache. One-time cost
    per dataset (~minutes on GPU); subsequent retrieve_one calls are O(1)."""
    from .embedding import _emb_path
    emb_cache_dir = Path(cfg["cache"]["embeddings_text"])
    facts = set()
    for cid in chunk_ids:
        p = Path(triples_dir) / f"{cid}.json"
        if not p.exists(): continue
        for t in json.load(open(p)).get("triples", []) or []:
            if not isinstance(t, (list, tuple)) or len(t) < 3: continue
            s, pr, o = t[0], t[1], t[2]
            if not (isinstance(s, str) and isinstance(pr, str) and isinstance(o, str)):
                continue
            facts.add(f"{s} {pr} {o}")
    if not facts: return

    miss = [f for f in facts
              if not _emb_path(f, "query_to_fact", emb_cache_dir).exists()]
    if verbose:
        print(f"[cache:fact_emb] {len(facts):,} unique fact strings "
                f"({len(miss):,} missing) …", flush=True)
    if miss:
        embed_texts(miss, emb_cache_dir, "query_to_fact", batch_size=32)
        if verbose: print(f"[cache:fact_emb] done.", flush=True)
