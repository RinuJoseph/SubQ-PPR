"""Misc utilities: text normalization, hashing, JSON IO, config loader."""
import json
import hashlib
import re
import yaml
from pathlib import Path
from typing import Any


def text_processing(text: str) -> str:
    """HippoRAG-2's exact normalization
    (src/hipporag/utils/misc_utils.py:54).

    Lowercase, replace any non-[A-Za-z0-9-space] with space, strip.
    """
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[^A-Za-z0-9 ]", " ", text.lower()).strip()


def text_hash(text: str) -> str:
    """16-char SHA1 hex digest of text (for cache filenames)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_config(config_path, dataset_override: str = None) -> dict:
    """Load YAML config; resolve `__DATASET__` placeholders to the active
    dataset name; back-fill legacy `data` / `cache` / `output_dir` fields so
    older code can keep reading them.

    Args:
        config_path: path to YAML config.
        dataset_override: if given, replaces `dataset.name` from the file.
    """
    p = Path(config_path).resolve()
    raw = yaml.safe_load(open(p))

    name = dataset_override or raw.get("dataset", {}).get("name", "musique")
    raw.setdefault("dataset", {})["name"] = name

    # Resolve __DATASET__ placeholders throughout the dict
    def _resolve(node):
        if isinstance(node, str):
            return node.replace("__DATASET__", name)
        if isinstance(node, dict):
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(x) for x in node]
        return node
    cfg = _resolve(raw)

    # Back-compat: synthesise legacy `data`, `cache`, `output_dir` keys
    cache_root  = Path(cfg["paths"]["cache_root"])
    output_root = Path(cfg["paths"]["output_root"])
    dset_root   = Path(cfg["dataset"]["root"])
    cfg.setdefault("data", {})
    cfg["data"].setdefault("dataset_root",  str(dset_root))
    cfg["data"].setdefault("queries_json",  str(dset_root / f"{name}.json"))
    cfg["data"].setdefault("corpus_json",   str(dset_root / f"{name}_corpus.json"))
    # legacy field used by older callers — keep them pointing to the JSON corpus
    cfg["data"].setdefault("corpus_csv",    str(dset_root / f"{name}_corpus.json"))

    cfg.setdefault("cache", {})
    cfg["cache"].setdefault("passage_embeddings",
                              str(cache_root / "embeddings/passage"))
    cfg["cache"].setdefault("embeddings_text",
                              str(cache_root / "embeddings/text"))
    cfg["cache"].setdefault("ner_doc",         str(cache_root / "ner_doc"))
    cfg["cache"].setdefault("triples",         str(cache_root / "triples"))
    cfg["cache"].setdefault("decompositions",  str(cache_root / "decompositions"))
    cfg["data"]["decompositions"] = cfg["cache"]["decompositions"]

    cfg.setdefault("output_dir", str(output_root))

    cfg["_config_path"] = str(p)
    return cfg
