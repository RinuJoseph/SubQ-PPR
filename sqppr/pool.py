"""Per-query pool construction.

Pool = mq top-K1 ∪ each sub-q top-K2, deduplicated. Order: mq's chunks first
(by mq rank), then dedup-append each sub-q's chunks in sub-q order.
"""
from pathlib import Path
from typing import Tuple

import numpy as np

from .embedding import embed_text, min_max_normalize


def build_pool(qr: dict, passage_embs: np.ndarray, passage_keys: list,
                 emb_cache_dir: Path,
                 mq_top: int = 50, sq_top: int = 10) -> Tuple[list, dict]:
    """Returns (pool, mq_score_of_full):

    pool:                list of chunk_ids in pool order (mq first, then rescues)
    mq_score_of_full:    dict cid -> mq DPR score (every corpus chunk)
    """
    mq_emb = embed_text(qr["query"], emb_cache_dir, "query_to_passage")
    mq_scores = min_max_normalize(np.dot(passage_embs, mq_emb.T).squeeze())
    mq_order = np.argsort(mq_scores)[::-1]
    mq_pool = [passage_keys[i] for i in mq_order[:mq_top]]
    seen = set(mq_pool)
    pool = list(mq_pool)
    mq_score_of_full = {passage_keys[i]: float(mq_scores[i])
                          for i in range(len(passage_keys))}

    for sub in qr.get("llm_decomposition", []) or []:
        sq_emb = embed_text(sub, emb_cache_dir, "query_to_passage")
        sq_scores = min_max_normalize(np.dot(passage_embs, sq_emb.T).squeeze())
        sq_order = np.argsort(sq_scores)[::-1][:sq_top]
        for idx in sq_order:
            cid = passage_keys[idx]
            if cid in seen: continue
            seen.add(cid)
            pool.append(cid)

    return pool, mq_score_of_full
