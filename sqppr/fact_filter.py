"""gpt-4o-mini fact filter.

For a query, score every pool triple by cosine to the main query, take top-K
candidates (default 30), then ask the LLM which contain evidence for the main
question or any sub-question. Returns a list of triple indices the LLM kept.
"""
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from openai import OpenAI

import sys
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from prompts.fact_filter import build_messages, FACT_FILTER_SYSTEM
from .embedding import embed_text, min_max_normalize

_OPENAI_CLIENT = None


def _client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


def top_k_candidate_facts(triples: list, fact_embs: np.ndarray,
                            main_query: str, emb_cache_dir: Path,
                            input_top_k: int = 30) -> List[Tuple[int, float, tuple]]:
    """Top-K facts ranked by cosine to main_query.
    Returns list of (triple_idx, score, (subject, predicate, object))."""
    if not triples:
        return []
    q_emb = embed_text(main_query, emb_cache_dir, "query_to_fact")
    scores = np.dot(fact_embs, q_emb.T)
    scores = np.squeeze(scores) if scores.ndim == 2 else scores
    scores = min_max_normalize(scores)
    K = min(input_top_k, len(triples))
    top_idx = np.argsort(scores)[::-1][:K]
    return [(int(i), float(scores[i]),
             (triples[i]["subject"], triples[i]["predicate"], triples[i]["object"]))
            for i in top_idx]


def parse_filter_response(text: str, max_n: int) -> List[int]:
    """Parse the LLM's comma-separated index list. Returns 0-based indices."""
    if not text or "NONE" in text.upper():
        return []
    nums = re.findall(r"\d+", text)
    out, seen = [], set()
    for s in nums:
        try: n = int(s)
        except ValueError: continue
        if 1 <= n <= max_n and n not in seen:
            seen.add(n); out.append(n - 1)
    return out


def llm_filter_facts(main_query: str, sub_questions: list,
                      candidate_facts: list,
                      model: str = "gpt-4o-mini",
                      temperature: float = 0.0,
                      max_completion_tokens: int = 120) -> List[int]:
    """Call the LLM filter. Returns 0-based indices into candidate_facts that
    the LLM kept. Empty list if LLM says NONE or returns nothing."""
    if not candidate_facts:
        return []
    messages = build_messages(main_query, sub_questions, candidate_facts)
    resp = _client().chat.completions.create(
        model=model, messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    text = resp.choices[0].message.content or ""
    return parse_filter_response(text, len(candidate_facts))


def select_facts(main_query: str, sub_questions: list,
                   triples: list, fact_embs: np.ndarray,
                   emb_cache_dir: Path,
                   input_top_k: int = 30,
                   output_cap_k=None,
                   fallback_top_k: int = 5,
                   model: str = "gpt-4o-mini") -> List[int]:
    """End-to-end fact selection:
    1. Take top-`input_top_k` candidates by cosine.
    2. LLM filter (gpt-4o-mini, open-ended prompt).
    3. Hard-cap kept count at `output_cap_k` if set (None = open / no cap).
    4. If LLM keeps 0, fall back to top-`fallback_top_k` by cosine.

    Returns triple indices (into `triples`) used to seed the PPR entity reset.
    """
    if not triples:
        return []
    cands = top_k_candidate_facts(triples, fact_embs, main_query,
                                    emb_cache_dir, input_top_k)
    kept_local = llm_filter_facts(main_query, sub_questions, cands, model=model)
    if kept_local:
        if output_cap_k is not None:
            kept_local = kept_local[:output_cap_k]
        return [cands[i][0] for i in kept_local if i < len(cands)]
    # Fallback: top-K by cosine
    return [c[0] for c in cands[:fallback_top_k]]
