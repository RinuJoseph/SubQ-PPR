"""NV-Embed v2 wrapper with disk cache.

Two caches:
  - cache/embeddings/passage/  passage_embeddings.npy + passage_keys.json
    (one big matrix, built once by build_passage_embeddings).
  - cache/embeddings/text/<sha1>.npy
    (per-text query/sub-q/fact embeddings, built on demand).

Standalone: builds everything from corpus + NV-Embed-v2. No reliance on any
external embedding cache.
"""
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .utils import text_hash

HIPPO_ROOT = Path("/workspace/storage/GraphRAG/HippoRAG")
_CACHE: dict = {}
_BACKEND_LOCK = threading.Lock()
# NV-Embed-v2's batch_encode is NOT thread-safe — multiple workers
# colliding on the GPU forward pass silently crashes. Serialise GPU work
# under this lock; LLM-filter calls (network-bound) still parallel.
_ENCODE_LOCK  = threading.Lock()


def _get_backend(model_name: str = "nvidia/NV-Embed-v2"):
    """Load NV-Embed-v2 via HippoRAG-2's NV-Embed wrapper.

    Thread-safe singleton: a Lock prevents 16 workers from each loading the
    7 GB model concurrently (which thrashed GPU memory and stalled). The
    fast-path (model already loaded) does NOT take the lock.
    """
    if "hr" in _CACHE:
        return _CACHE["hr"]
    with _BACKEND_LOCK:
        if "hr" in _CACHE:        # double-checked locking
            return _CACHE["hr"]
        sys.path.insert(0, str(HIPPO_ROOT))
        from src.hipporag.HippoRAG import HippoRAG
        from src.hipporag.utils.config_utils import BaseConfig
        cfg = BaseConfig(
            save_dir="outputs/musique",
            llm_base_url="https://api.openai.com/v1",
            llm_name="gpt-4o-mini", dataset="musique",
            embedding_model_name=model_name,
            force_index_from_scratch=False,
            force_openie_from_scratch=False,
            retrieval_top_k=200, linking_top_k=5,
            max_qa_steps=3, qa_top_k=5,
            graph_type="facts_and_sim_passage_node_unidirectional",
            embedding_batch_size=8, max_new_tokens=None,
            corpus_len=0, openie_mode="online",
            damping=0.5,
        )
        import os; os.chdir(HIPPO_ROOT)
        hr = HippoRAG(global_config=cfg)
        # NOTE: do NOT call hr.prepare_retrieval_objects() — that loads
        # HippoRAG's own passage cache. We only need the encoder.
        _CACHE["hr"] = hr
        return hr


# ---------------------------------------------------------------------------
# Per-text cache (queries, sub-queries, facts)
# ---------------------------------------------------------------------------
def _emb_path(text: str, kind: str, cache_dir: Path) -> Path:
    return cache_dir / f"{text_hash(f'{kind}:{text}')}.npy"


def min_max_normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def embed_text(text: str, cache_dir: Path,
                instruction_kind: str = "query_to_passage") -> np.ndarray:
    p = _emb_path(text, instruction_kind, cache_dir)
    if p.exists():
        return np.load(p)
    return embed_texts([text], cache_dir, instruction_kind)[0]


def embed_texts(texts: List[str], cache_dir: Path,
                  instruction_kind: str = "query_to_passage",
                  batch_size: int = 32) -> np.ndarray:
    """Batched NV-Embed encoding with per-text disk cache.
    Used for queries / sub-questions / facts (instruction-conditioned)."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    out = [None] * n
    todo, todo_idx = [], []
    for i, t in enumerate(texts):
        p = _emb_path(t, instruction_kind, cache_dir)
        if p.exists():
            out[i] = np.load(p)
        else:
            todo.append(t); todo_idx.append(i)
    if todo:
        hr = _get_backend()
        sys.path.insert(0, str(HIPPO_ROOT))
        from src.hipporag.prompts.linking import get_query_instruction
        instr = get_query_instruction(instruction_kind)
        for k in range(0, len(todo), batch_size):
            chunk = todo[k:k + batch_size]
            with _ENCODE_LOCK:
                embs = hr.embedding_model.batch_encode(chunk, instruction=instr, norm=True)
            embs = np.asarray(embs, dtype=np.float32)
            if embs.ndim == 1:
                embs = embs[None, :]
            for offset, t in enumerate(chunk):
                emb = embs[offset]
                out[todo_idx[k + offset]] = emb
                np.save(_emb_path(t, instruction_kind, cache_dir), emb)
    return np.stack(out)


# ---------------------------------------------------------------------------
# Passage embeddings (corpus index)
# ---------------------------------------------------------------------------
def _passage_paths(cache_dir: Path) -> Tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    return cache_dir / "passage_embeddings.npy", cache_dir / "passage_keys.json"


def embed_passages(corpus: Dict[str, dict], cache_dir: Path,
                     batch_size: int = 8) -> Tuple[np.ndarray, List[str]]:
    """Encode every passage in the corpus and write a single (N, D) NPY matrix
    + a JSON list of chunk_ids alongside it.

    Passages are encoded with NO instruction (NV-Embed-v2 default for docs).
    Embeddings are L2-normalized (norm=True) so dot product == cosine.

    Idempotent: if cached matrix matches the corpus key list, reuse it.
    """
    npy_p, keys_p = _passage_paths(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_ids = list(corpus.keys())
    if npy_p.exists() and keys_p.exists():
        existing_keys = json.load(open(keys_p))
        if existing_keys == chunk_ids:
            return np.load(npy_p), existing_keys
        print("[passage_emb] cache key order differs from corpus — rebuilding.")

    hr = _get_backend()
    n = len(chunk_ids)
    print(f"[passage_emb] encoding {n} passages (NV-Embed-v2, batch={batch_size}) …")
    out = None
    t0 = time.time()
    for i in range(0, n, batch_size):
        chunk = chunk_ids[i:i + batch_size]
        texts = [corpus[c]["content"] for c in chunk]
        with _ENCODE_LOCK:
            embs = hr.embedding_model.batch_encode(texts, norm=True)
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1: embs = embs[None, :]
        if out is None:
            out = np.zeros((n, embs.shape[1]), dtype=np.float32)
        out[i:i + len(chunk)] = embs
        if (i // batch_size) % 50 == 0 or i + batch_size >= n:
            done = min(i + batch_size, n)
            rate = done / max(1, time.time() - t0)
            print(f"  [{done:>5d}/{n}] {time.time()-t0:.0f}s  ({rate:.1f}/s)",
                    flush=True)
    np.save(npy_p, out)
    json.dump(chunk_ids, open(keys_p, "w"))
    print(f"[passage_emb] saved -> {npy_p} ({out.nbytes / 1e6:.1f} MB)")
    return out, chunk_ids


def _bootstrap_from_hipporag_parquet(corpus: Dict[str, dict], cache_dir: Path
                                       ) -> Tuple[np.ndarray, List[str]]:
    """One-time fast path: if HippoRAG-2 has already encoded a corpus that
    matches ours and saved it as vdb_chunk.parquet, reuse those embeddings.

    The cache_dir path looks like:  .../cache/<dataset>/embeddings/passage
    so we infer the dataset name and look for HippoRAG-2's pre-built parquet
    at outputs/<dataset>/gpt-4o-mini_nvidia_NV-Embed-v2/chunk_embeddings/.
    Falls back to a fresh GPU encode if no matching parquet exists or any
    corpus passage is missing from it.
    """
    cache_dir = Path(cache_dir)
    dataset_name = None
    for part in reversed(cache_dir.parts):
        if part not in ("passage", "embeddings"):
            dataset_name = part; break
    if not dataset_name:
        return None
    # Candidate parquet locations. Order: HippoRAG-2 primary outputs, then
    # alt LLM dirs (still NV-Embed-v2). We do NOT accept text-embedding-3-*
    # parquets — those vectors live in a different space.
    candidates = [
        Path("/workspace/storage/GraphRAG/HippoRAG/outputs/"
              f"{dataset_name}/gpt-4o-mini_nvidia_NV-Embed-v2/"
              "chunk_embeddings/vdb_chunk.parquet"),
    ]
    # Fallback: outputs_llama/<dataset>/<llm>_nvidia_NV-Embed-v2/.../vdb_chunk.parquet
    for base in (Path("/workspace/storage/GraphRAG/HippoRAG/outputs_llama"),
                   Path("/workspace/storage/GraphRAG/HippoRAG/outputs_llama_v1")):
        if not base.exists(): continue
        for sub in base.glob(f"{dataset_name}/*_nvidia_NV-Embed-v2/chunk_embeddings/vdb_chunk.parquet"):
            candidates.append(sub)

    parquet = next((p for p in candidates if p.exists()), None)
    if parquet is None:
        print(f"[passage_emb] no NV-Embed-v2 parquet for {dataset_name!r} "
                f"-> will encode from scratch")
        return None
    try:
        import pandas as pd
    except ImportError:
        return None
    print(f"[passage_emb] bootstrapping from HippoRAG-2 parquet:")
    print(f"  {parquet}")
    df = pd.read_parquet(parquet)
    # df has rows keyed by content hash. Map back to corpus chunk_ids by
    # matching the 'content' column.
    if "content" not in df.columns:
        print(f"  [skip] no 'content' column found, expected: {list(df.columns)}")
        return None
    content_to_emb = {row["content"]: row["embedding"]
                        for _, row in df[["content", "embedding"]].iterrows()}
    chunk_ids = list(corpus.keys())
    miss = 0
    embs = []
    for cid in chunk_ids:
        e = content_to_emb.get(corpus[cid]["content"])
        if e is None: miss += 1; embs.append(None)
        else: embs.append(np.asarray(e, dtype=np.float32))
    if miss > 0:
        print(f"  [skip] {miss}/{len(chunk_ids)} passages missing from parquet — "
                f"falling back to fresh encode")
        return None
    out = np.stack(embs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_p, keys_p = _passage_paths(cache_dir)
    np.save(npy_p, out)
    json.dump(chunk_ids, open(keys_p, "w"))
    print(f"  saved -> {npy_p}  ({out.nbytes/1e6:.1f} MB, {out.shape})")
    return out, chunk_ids


def passage_embeddings_and_keys(cache_dir: Path = None,
                                  corpus: Dict[str, dict] = None
                                  ) -> Tuple[np.ndarray, List[str]]:
    """Load passage embeddings from local cache.
    If absent: try bootstrapping from HippoRAG-2's pre-built parquet, else
    build from scratch via NV-Embed (requires `corpus`)."""
    if cache_dir is None:
        raise ValueError("cache_dir is required (path to passage embeddings cache).")
    cache_dir = Path(cache_dir)
    npy_p, keys_p = _passage_paths(cache_dir)
    if npy_p.exists() and keys_p.exists():
        return np.load(npy_p), json.load(open(keys_p))
    if corpus is None:
        raise FileNotFoundError(
            f"Passage embeddings cache missing at {cache_dir}. "
            f"Run `python scripts/build_cache.py --config configs/default.yaml` first.")
    bootstrapped = _bootstrap_from_hipporag_parquet(corpus, cache_dir)
    if bootstrapped is not None:
        return bootstrapped
    return embed_passages(corpus, cache_dir)
