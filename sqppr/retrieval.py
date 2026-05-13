"""End-to-end retrieval for a single query.

Composes pool → graph → fact filter → PPR → ranking.
"""
from pathlib import Path
from typing import Dict

import numpy as np

from .embedding import embed_texts
from .pool import build_pool
from .graph import collect_triples_and_entities, build_graph
from .fact_filter import select_facts
from .ppr import (
    build_reset, run_ppr,
    passage_scores_own_only, passage_scores_own_plus_nbr,
)


def retrieve_one(qr: dict, corpus: dict,
                   passage_embs: np.ndarray, passage_keys: list,
                   cfg: dict) -> dict:
    """Run the full retrieval pipeline on one query record.

    Returns a dict with: pool, ranking (sorted chunk_ids), top_5 (with metadata),
    final_pool_coverage, graph_stats, kept_triple_indices.
    """
    emb_cache_dir   = Path(cfg["cache"]["embeddings_text"])
    triples_dir     = Path(cfg["cache"]["triples"])

    # 1. Pool
    pool, mq_score_of_full = build_pool(
        qr, passage_embs, passage_keys, emb_cache_dir,
        mq_top=cfg["pool"]["mq_top"], sq_top=cfg["pool"]["sq_top"],
    )

    # 2. Graph
    triples, chunk_triple_entities, all_entities = \
        collect_triples_and_entities(pool, triples_dir)
    G, name_to_idx, names = build_graph(
        pool, triples, chunk_triple_entities, all_entities,
        emb_cache_dir, synonym_threshold=cfg["graph"]["synonym_threshold"],
    )

    # 3. Fact embeddings + LLM filter
    fact_embs = None
    kept_indices = []
    if triples:
        f_strs = [f"{t['subject']} {t['predicate']} {t['object']}"
                  for t in triples]
        fact_embs = embed_texts(f_strs, emb_cache_dir, "query_to_fact")
        kept_indices = select_facts(
            qr["query"], qr.get("llm_decomposition", []) or [],
            triples, fact_embs, emb_cache_dir,
            input_top_k=cfg["fact_filter"]["input_top_k"],
            output_cap_k=cfg["fact_filter"].get("output_cap_k"),
            fallback_top_k=cfg["fact_filter"]["fallback_top_k"],
            model=cfg["llm"]["filter_model"],
        )

    # 4. PPR reset + run
    node_w = build_reset(
        qr["query"], pool, mq_score_of_full, triples, all_entities,
        kept_indices, fact_embs, name_to_idx, emb_cache_dir,
        passage_node_weight=cfg["ppr"]["passage_node_weight"],
        fallback_top_k=cfg["fact_filter"]["fallback_top_k"],
    )
    pr = run_ppr(G, node_w, damping=cfg["ppr"]["damping"])

    # 5. Score passages
    if cfg["ppr"]["scoring_rule"] == "own_plus_nbr":
        scores = passage_scores_own_plus_nbr(pool, pr, G, names, name_to_idx)
    else:
        scores = passage_scores_own_only(pool, pr, name_to_idx)

    ranking = sorted(pool, key=lambda c: -scores[c])
    gold = [g["chunk_id"] for g in qr["gold"] if g.get("chunk_id")]
    gold_set = set(gold)
    pool_set = set(pool)
    coverage = (sum(1 for c in gold if c in pool_set) / len(gold)) if gold else None

    return {
        "qid":              qr["qid"],
        "query":            qr["query"],
        "n_sub_questions":  len(qr.get("llm_decomposition", []) or []),
        "pool":             pool,
        "n_pool":           len(pool),
        "n_entity_nodes":   len(all_entities),
        "n_edges":          G.ecount(),
        "n_triples":        len(triples),
        "n_facts_kept_by_llm": len(kept_indices),
        "ranking":          ranking,
        "scores":           scores,
        "gold":             qr["gold"],
        "n_gold":           len(gold),
        "final_pool_coverage": coverage,
        "top_5": [
            {
                "rank": r + 1, "chunk_id": cid,
                "doc_id": corpus[cid]["doc_id"],
                "title":  corpus[cid]["title"],
                "score":  round(scores[cid], 8),
                "is_gold": cid in gold_set,
            }
            for r, cid in enumerate(ranking[:5])
        ],
        "gold_landings": [
            {
                "hop": g["hop"], "chunk_id": g.get("chunk_id"),
                "answer": g.get("answer"),
                "in_pool": g.get("chunk_id") in pool_set if g.get("chunk_id") else False,
                "rank": (ranking.index(g["chunk_id"]) + 1)
                          if g.get("chunk_id") in pool_set else None,
            }
            for g in qr["gold"]
        ],
    }
