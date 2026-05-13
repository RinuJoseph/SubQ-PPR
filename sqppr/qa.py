"""Final QA using HippoRAG-2's rag_qa_musique one-shot prompt + gpt-4o-mini."""
import re
import sys
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from prompts.rag_qa_musique import build_messages

_OPENAI_CLIENT = None


def _client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


def parse_answer(response: str) -> str:
    """Extract text after 'Answer:' marker (case-insensitive)."""
    if response is None: return ""
    m = re.search(r"answer\s*:\s*(.+?)(?:$|\n)",
                   response, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def qa_one(question: str, top_passages: List[dict], corpus: Dict[str, dict],
             model: str = "gpt-4o-mini",
             temperature: float = 0.0,
             max_completion_tokens: int = 400) -> dict:
    """Run QA on one query with its top-K passages.

    top_passages: list of {chunk_id, title, ...} (we'll fetch text from corpus)
    Returns: {raw_response, predicted_answer}
    """
    passages = [
        {"title": corpus[p["chunk_id"]]["title"],
         "text":  corpus[p["chunk_id"]]["text"]}
        for p in top_passages
    ]
    messages = build_messages(passages, question)
    resp = _client().chat.completions.create(
        model=model, messages=messages,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    raw = resp.choices[0].message.content or ""
    return {
        "raw_response":     raw,
        "predicted_answer": parse_answer(raw),
    }
