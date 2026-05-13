"""Corpus + query loading.

Generic over datasets: dispatches to per-dataset adapters in
`sqppr/datasets.py`. Adds the per-qid LLM decomposition (gpt-4o-mini) from
the local cache dir on top of whatever the adapter returned.

`load_corpus()` / `load_queries()` are kept for backward compatibility but now
take a dataset name; new code should use `load_dataset_with_decomp()`.
"""
import json
from pathlib import Path
from typing import Dict, List, Union

from .datasets import load_dataset, dataset_paths, SUPPORTED


# ---------------------------------------------------------------------------
# Decomposition cache loader
# ---------------------------------------------------------------------------
def _load_decompositions(source: Path) -> Dict[str, List[str]]:
    """Accept either:
       - a directory of per-qid JSONs (cache_dir layout)
           each file: {"qid":..., "sub_questions": [...]}
       - a single JSON list  [{"qid":..., "llm_decomposition": [...]}, ...]
           (legacy summary file)
    Returns dict qid -> list of stripped sub-question strings."""
    source = Path(source)
    out: Dict[str, List[str]] = {}
    if source.is_dir():
        for p in source.glob("*.json"):
            rec = json.load(open(p))
            qid = rec["qid"]
            raw = rec.get("sub_questions") or rec.get("llm_decomposition") or []
            out[qid] = [l.split(":", 1)[1].strip() if ":" in l else l.strip()
                          for l in raw]
        return out
    if source.exists():
        rows = json.load(open(source))
        for r in rows:
            raw = r.get("llm_decomposition") or r.get("sub_questions") or []
            out[r["qid"]] = [l.split(":", 1)[1].strip() if ":" in l else l.strip()
                              for l in raw]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_dataset_with_decomp(
    dataset_name: str,
    dataset_root: Path,
    decompositions_dir: Path,
) -> (dict, list):
    """Load `(corpus, queries)` for one dataset, attaching `llm_decomposition`
    (from the per-qid cache) to each query record."""
    corpus, queries = load_dataset(dataset_name, dataset_root)
    decomp = _load_decompositions(Path(decompositions_dir))
    for q in queries:
        q["llm_decomposition"] = decomp.get(q["qid"], [])
    return corpus, queries


# ---- Back-compat shims (used by older scripts that still pass paths) ----
def load_corpus(corpus_csv: Path) -> Dict[str, dict]:
    """Back-compat. If `corpus_csv` is the old MuSiQue passages.csv, parse it;
    if it's a JSON corpus file, route through the new path. Prefer
    `load_dataset_with_decomp(dataset_name, ...)` in new code."""
    p = Path(corpus_csv)
    if p.suffix == ".csv":
        import csv
        out: Dict[str, dict] = {}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                cid = row["chunk_id"]
                out[cid] = {
                    "doc_id": row["doc_id"], "title": row["title"],
                    "text":   row["text"],
                    "content": f"{row['title']}\n{row['text']}",
                }
        return out
    # JSON corpus
    from .datasets import _load_corpus_json
    return _load_corpus_json(p)[0]


def load_queries(queries_json: Path, decompositions_src: Union[Path, str],
                   corpus: Dict[str, dict]) -> List[dict]:
    """Back-compat. Routes to MuSiQue adapter if queries_json is a MuSiQue
    sample list (detects via `question_decomposition` key on row[0])."""
    samples = json.load(open(queries_json))
    if not samples:
        return []
    decomp = _load_decompositions(Path(decompositions_src))
    title_to_chunks: Dict[str, List[str]] = {}
    for cid, row in corpus.items():
        title_to_chunks.setdefault(row["title"], []).append(cid)
    if "question_decomposition" in samples[0]:
        from .datasets import _adapter_musique
        queries = _adapter_musique(samples, corpus, title_to_chunks)
    elif "claim" in samples[0]:
        from .datasets import _adapter_hover
        queries = _adapter_hover(samples, corpus, title_to_chunks)
    else:
        from .datasets import _adapter_hotpotqa
        queries = _adapter_hotpotqa(samples, corpus, title_to_chunks)
    for q in queries:
        q["llm_decomposition"] = decomp.get(q["qid"], [])
    return queries
