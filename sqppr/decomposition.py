"""Query decomposition via LLM.

Default model: gpt-4o-mini (OpenAI). To reproduce the paper's headline numbers,
use claude-opus-4-7 — pass `decomposition.model: claude-opus-4-7` in config and
set ANTHROPIC_API_KEY.

Decomposition for each qid is cached as one JSON file at
`cache/decompositions/<qid>.json`:

  {"qid": <id>, "model": <name>, "raw_response": <str>,
   "sub_questions": ["Q1 text", "Q2 text", ...]}

Re-running is idempotent: existing cache files are reused.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from prompts.query_decomposition import build_messages

_SUBQ_RE = re.compile(r"\s*Q(\d+)\s*:\s*(.+?)\s*$")


def _parse_subqs(raw: str) -> List[str]:
    out = []
    for line in raw.splitlines():
        m = _SUBQ_RE.match(line)
        if m:
            out.append(m.group(2).strip())
    return out


def _call_llm(model: str, messages, max_tokens: int = 600, temperature: float = 0):
    """Route by model name prefix: 'claude*' -> Anthropic, else OpenAI."""
    if model.lower().startswith("claude"):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, messages=messages,
        )
        return resp.content[0].text
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def decompose_one(qid: str, query: str, model: str, cache_dir: Path) -> dict:
    """Decompose one query; cache hit short-circuits the LLM call."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{qid}.json"
    if p.exists():
        return json.load(open(p))
    raw = _call_llm(model, build_messages(query)).strip()
    rec = {
        "qid":           qid,
        "query":         query,
        "model":         model,
        "raw_response":  raw,
        "sub_questions": _parse_subqs(raw),
    }
    json.dump(rec, open(p, "w"), indent=2, ensure_ascii=False)
    return rec


def load_decomposition(qid: str, cache_dir: Path) -> List[str]:
    """Return cached sub-questions for qid, or [] if not cached yet."""
    p = Path(cache_dir) / f"{qid}.json"
    if not p.exists():
        return []
    return json.load(open(p)).get("sub_questions", [])


def build_all(queries: List[Dict], model: str, cache_dir: Path,
                workers: int = 8) -> Dict[str, dict]:
    """Decompose every query in `queries` (skipping any already cached).
    Returns dict qid -> record."""
    cache_dir = Path(cache_dir)
    todo = [(q["qid"], q["query"]) for q in queries
            if not (cache_dir / f"{q['qid']}.json").exists()]
    results: Dict[str, dict] = {}
    # Load existing cache
    for q in queries:
        p = cache_dir / f"{q['qid']}.json"
        if p.exists():
            results[q["qid"]] = json.load(open(p))
    if not todo:
        return results
    print(f"[decompose] {len(todo)} queries to decompose with {model}  "
            f"(workers={workers})")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(decompose_one, qid, qtext, model, cache_dir): qid
                  for qid, qtext in todo}
        n_done = n_err = 0
        for fut in as_completed(futs):
            qid = futs[fut]
            try:
                results[qid] = fut.result()
            except Exception as e:
                n_err += 1
                if n_err < 5: print(f"  [err] {qid}: {e}", flush=True)
            n_done += 1
            if n_done % 50 == 0 or n_done == len(todo):
                print(f"  [{n_done:4d}/{len(todo)}]  errs={n_err}", flush=True)
    return results
