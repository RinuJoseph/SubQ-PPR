"""NER + OpenIE wrapper.

Wraps HippoRAG-2's `OpenIE` class so we use their exact prompts and parsing.
Caches per-chunk NER and triples on disk so we run the LLM at most once
per (chunk, type).
"""
import sys
from pathlib import Path
from typing import List, Tuple

from .utils import save_json, load_json, text_hash


def _get_openie_backend():
    """Lazy-load HippoRAG-2's OpenIE class with their gpt-4o-mini wrapper."""
    HIPPO_ROOT = Path("/workspace/storage/GraphRAG/HippoRAG")
    sys.path.insert(0, str(HIPPO_ROOT))
    from src.hipporag.information_extraction.openie_openai import OpenIE
    from .embedding import _get_backend
    hr = _get_backend()
    return OpenIE(hr.llm_model)


def ner_chunk(oie, chunk_id: str, content: str, cache_dir: Path) -> dict:
    """Run NER on a passage. Returns the parsed result and caches it.

    Skips the LLM call if the cache entry already exists.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{chunk_id}.json"
    if p.exists():
        return load_json(p)
    r = oie.ner(chunk_key=chunk_id, passage=content)
    result = {
        "chunk_id":        r.chunk_id,
        "unique_entities": list(r.unique_entities) if r.unique_entities else [],
        "raw_response":    r.response,
        "metadata":        r.metadata,
    }
    save_json(result, p)
    return result


def ner_text(oie, text: str, cache_dir: Path) -> dict:
    """NER on free-form text (a query or sub-question). Cached by text-hash."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = text_hash(text)
    p = cache_dir / f"{key}.json"
    if p.exists():
        return load_json(p)
    r = oie.ner(chunk_key=key, passage=text)
    result = {
        "text":            text,
        "label":           key,
        "unique_entities": list(r.unique_entities) if r.unique_entities else [],
        "raw_response":    r.response,
        "metadata":        r.metadata,
    }
    save_json(result, p)
    return result


def triples_chunk(oie, chunk_id: str, content: str,
                    named_entities: List[str], cache_dir: Path) -> dict:
    """OpenIE on a passage, conditioned on its NER list. Cached per chunk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{chunk_id}.json"
    if p.exists():
        return load_json(p)
    r = oie.triple_extraction(
        chunk_key=chunk_id, passage=content, named_entities=named_entities,
    )
    result = {
        "chunk_id":             r.chunk_id,
        "named_entities_input": named_entities,
        "triples":              [list(t) for t in r.triples] if r.triples else [],
        "raw_response":          r.response,
        "metadata":              r.metadata,
    }
    save_json(result, p)
    return result


def load_chunk_triples(chunk_id: str, cache_dir: Path) -> list:
    """Read cached triples for a chunk. Returns list of [s, p, o]."""
    p = Path(cache_dir) / f"{chunk_id}.json"
    if not p.exists():
        return []
    return load_json(p).get("triples", [])


def load_chunk_ner(chunk_id: str, cache_dir: Path) -> list:
    p = Path(cache_dir) / f"{chunk_id}.json"
    if not p.exists():
        return []
    return load_json(p).get("unique_entities", [])
