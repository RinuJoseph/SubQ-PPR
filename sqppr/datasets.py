"""Per-dataset adapters.

Each adapter turns a (queries.json, corpus.json) pair into the uniform record:

    corpus: dict[chunk_id -> {doc_id, title, text, content}]
    queries: list[{qid, query, gold, sample}]
       gold: list[{hop, chunk_id, answer}]    # chunk_id resolved against corpus

`chunk_id` is the corpus's own integer id rendered as `"chunk-<id>"` so the
filesystem-as-cache layout (one JSON per chunk_id) works on every dataset.

Dispatch via `load_dataset(name, dataset_root)`. Supported names:

    musique  hotpotqa  2wikimultihopqa  hover  popqa  nq_rear

`question` is taken from the natural-language query field of each dataset
(question / claim / etc.). `gold` is resolved against the SEPARATE
`*_corpus.json` corpus file (not the inline context lists), so the chunk_id
space matches NV-Embed indexing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _chunk_id(title: str, text: str) -> str:
    """HippoRAG-2's exact chunk_id scheme: 'chunk-' + md5(title+"\\n"+text).
    Stable across runs and matches the legacy passages.csv cache layout.
    """
    h = hashlib.md5(f"{title}\n{text}".encode("utf-8")).hexdigest()
    return f"chunk-{h}"


# ---------------------------------------------------------------------------
# Corpus loader (uniform across datasets)
# ---------------------------------------------------------------------------
def _load_corpus_json(corpus_json: Path) -> Tuple[Dict[str, dict], Dict[str, List[str]]]:
    """Read a JSON corpus file. Format: list of {id, title, text}.

    Returns:
      corpus: chunk_id ("chunk-<id>") -> {doc_id, title, text, content}
      title_to_chunks: title -> [chunk_id, ...]  (ordered by appearance; needed
                                                    for supporting_facts-style
                                                    gold resolution which keys
                                                    on title alone)
    """
    rows = json.load(open(corpus_json))
    corpus: Dict[str, dict] = {}
    title_to_chunks: Dict[str, List[str]] = {}
    for i, row in enumerate(rows):
        title = row["title"]; text = row["text"]
        # Use HippoRAG-2's chunk_id scheme = chunk-md5(title+"\n"+text). This
        # is stable across datasets and matches the existing on-disk caches
        # built off HippoRAG-2's passages.csv.
        cid = _chunk_id(title, text)
        raw_id = row.get("id", row.get("idx", i))
        corpus[cid] = {
            "doc_id":  str(raw_id),
            "title":   title,
            "text":    text,
            "content": f"{title}\n{text}",
        }
        title_to_chunks.setdefault(title, []).append(cid)
    return corpus, title_to_chunks


# ---------------------------------------------------------------------------
# Per-dataset query+gold adapters
# ---------------------------------------------------------------------------
def _adapter_musique(samples, corpus, title_to_chunks):
    """gold = paragraphs[support_idx]; match title+text against corpus content."""
    content_to_chunk = {v["content"]: k for k, v in corpus.items()}
    out = []
    for s in samples:
        paras = {p["idx"]: p for p in s["paragraphs"]}
        gold = []
        for i, qd in enumerate(s.get("question_decomposition", []), start=1):
            sup = qd.get("paragraph_support_idx")
            cid = None
            if sup is not None and sup in paras:
                p = paras[sup]
                key = f"{p['title']}\n{p.get('paragraph_text', p.get('text', ''))}"
                cid = content_to_chunk.get(key)
                if cid is None:
                    # fall back to title-only match
                    cands = title_to_chunks.get(p["title"], [])
                    if cands: cid = cands[0]
            gold.append({"hop": i, "chunk_id": cid,
                            "subq": qd.get("question"),
                            "answer": qd.get("answer")})
        out.append({
            "qid":    s["id"],
            "query":  s["question"],
            "gold":   gold,
            "answer": s.get("answer"),
            "sample": s,
        })
    return out


def _by_title_supporting(samples, corpus, title_to_chunks, qid_key, q_key,
                           answer_key="answer"):
    """Common adapter for HotpotQA / 2WikiMultiHopQA / HoVer — all three have
    `supporting_facts = [[title, sent_idx], ...]` keyed by title."""
    out = []
    for s in samples:
        sfs = s.get("supporting_facts", []) or []
        # Dedupe titles in original order
        seen = set(); titles = []
        for tup in sfs:
            if not tup: continue
            t = tup[0]
            if t in seen: continue
            seen.add(t); titles.append(t)
        gold = []
        for i, t in enumerate(titles, start=1):
            cands = title_to_chunks.get(t, [])
            cid = cands[0] if cands else None
            gold.append({"hop": i, "chunk_id": cid, "subq": None,
                            "answer": None, "title": t})
        out.append({
            "qid":    str(s[qid_key]),
            "query":  s[q_key],
            "gold":   gold,
            "answer": s.get(answer_key),
            "sample": s,
        })
    return out


def _adapter_hotpotqa(samples, corpus, title_to_chunks):
    return _by_title_supporting(samples, corpus, title_to_chunks,
                                  qid_key="_id", q_key="question")


def _adapter_2wikimultihopqa(samples, corpus, title_to_chunks):
    return _by_title_supporting(samples, corpus, title_to_chunks,
                                  qid_key="_id", q_key="question")


def _adapter_hover(samples, corpus, title_to_chunks):
    # HoVer's natural-language query is "claim". The 'question' field in
    # this dump is the same string. Use 'claim' canonically.
    return _by_title_supporting(samples, corpus, title_to_chunks,
                                  qid_key="hover_id", q_key="claim",
                                  answer_key="label")


def _adapter_popqa(samples, corpus, title_to_chunks):
    """popqa: gold = `paragraphs[].title` (treat every listed paragraph as gold
    support since they're all annotated relevant)."""
    out = []
    for s in samples:
        paras = s.get("paragraphs", []) or []
        gold = []
        for i, p in enumerate(paras, start=1):
            t = p.get("title")
            cands = title_to_chunks.get(t, [])
            cid = cands[0] if cands else None
            gold.append({"hop": i, "chunk_id": cid, "title": t,
                            "subq": None, "answer": None})
        out.append({
            "qid":    str(s["id"]),
            "query":  s["question"],
            "gold":   gold,
            "answer": s.get("obj") or (s.get("possible_answers") or [None])[0],
            "sample": s,
        })
    return out


def _adapter_nq_rear(samples, corpus, title_to_chunks):
    """nq_rear: contexts[] is a 10-passage mix of supporting + distractor
    paragraphs. Gold = only the contexts with `is_supporting == True`
    (mean ~3.8 per query). Earlier versions of this adapter incorrectly
    treated all 10 contexts as gold, which capped R@5 at 0.5."""
    out = []
    for i, s in enumerate(samples):
        ctxs = s.get("contexts", []) or []
        gold = []
        hop = 0
        for c in ctxs:
            if not c.get("is_supporting"): continue
            hop += 1
            t = c.get("title")
            cands = title_to_chunks.get(t, [])
            cid = cands[0] if cands else None
            gold.append({"hop": hop, "chunk_id": cid, "title": t,
                            "subq": None, "answer": None})
        out.append({
            "qid":    f"nq_rear_{i}",
            "query":  s["question"],
            "gold":   gold,
            "answer": (s.get("reference") or [None])[0],
            "sample": s,
        })
    return out


def _adapter_lveval(samples, corpus, title_to_chunks):
    """lveval: no chunk-level gold (gold is just an answer string).
    Pipeline runs end-to-end, recall@K reports None, QA F1 evaluates against
    `gold_ans` / first of `answers`."""
    out = []
    for s in samples:
        ans = s.get("gold_ans") or (s.get("answers") or [None])[0]
        out.append({
            "qid":    f"lveval_{s['id']}",
            "query":  s["question"],
            "gold":   [],
            "answer": ans,
            "sample": s,
        })
    return out


def _adapter_narrativeqa(samples, corpus, title_to_chunks):
    """narrativeqa: gold is document-level (one whole script per query).
    We don't compute chunk-level recall — leave gold empty. QA F1 uses the
    first acceptable answer."""
    out = []
    for i, s in enumerate(samples):
        ans = (s.get("answer") or [None])[0]
        doc = s.get("document", {})
        qid = doc.get("id", f"narrativeqa_{i}")
        out.append({
            "qid":    f"narrativeqa_{i}_{qid[:8]}",
            "query":  s["question"],
            "gold":   [],
            "answer": ans,
            "sample": s,
        })
    return out


_ADAPTERS = {
    "musique":                _adapter_musique,
    "hotpotqa":               _adapter_hotpotqa,
    "2wikimultihopqa":        _adapter_2wikimultihopqa,
    "hover":                  _adapter_hover,
    "popqa":                  _adapter_popqa,
    "nq_rear":                _adapter_nq_rear,
    "lveval":                 _adapter_lveval,
    "narrativeqa_dev_10_doc": _adapter_narrativeqa,
}


SUPPORTED = sorted(_ADAPTERS.keys())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def dataset_paths(name: str, root: Path) -> Tuple[Path, Path]:
    """Return (queries_json, corpus_json) for the given dataset name."""
    root = Path(root)
    return root / f"{name}.json", root / f"{name}_corpus.json"


def load_dataset(name: str, root: Path) -> Tuple[Dict[str, dict], List[dict]]:
    """Load (corpus, queries) for a named dataset.

    `queries[i]` always has fields: qid, query, gold (list of {hop, chunk_id,
    title, subq, answer}), answer, sample.
    """
    if name not in _ADAPTERS:
        raise ValueError(f"unknown dataset {name!r}. supported: {SUPPORTED}")
    q_path, c_path = dataset_paths(name, root)
    if not q_path.exists():
        raise FileNotFoundError(f"queries file missing: {q_path}")
    if not c_path.exists():
        raise FileNotFoundError(f"corpus file missing: {c_path}")
    corpus, title_to_chunks = _load_corpus_json(c_path)
    samples = json.load(open(q_path))
    queries = _ADAPTERS[name](samples, corpus, title_to_chunks)
    return corpus, queries
