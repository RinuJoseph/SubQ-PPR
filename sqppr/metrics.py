"""Evaluation metrics — F1, EM, Recall@K — using HippoRAG-2's exact functions.

normalize_answer: lowercase + remove punctuation + remove articles + collapse spaces
compute_f1: token-overlap precision-recall harmonic mean on normalized strings
compute_em: 1.0 if normalized strings match, else 0.0
recall_at_k: |gold ∩ retrieved[:k]| / |gold|
"""
import re
import string
from collections import Counter
from typing import List


def normalize_answer(text: str) -> str:
    """HippoRAG-2's exact normalize_answer
    (src/hipporag/utils/eval_utils.py:4)."""
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def collapse_spaces(t):  return " ".join(t.split())
    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)
    def lower(t): return t.lower()
    return collapse_spaces(remove_articles(remove_punc(lower(text))))


def compute_f1(predicted: str, gold: str) -> float:
    """HippoRAG-2's exact compute_f1
    (src/hipporag/evaluation/qa_eval.py:71-82)."""
    gold_tokens = normalize_answer(gold).split()
    pred_tokens = normalize_answer(predicted).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def compute_em(predicted: str, gold: str) -> float:
    return 1.0 if normalize_answer(predicted) == normalize_answer(gold) else 0.0


def recall_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    g = [x for x in gold if x]
    if not g: return None
    return sum(1 for x in g if x in set(retrieved[:k])) / len(g)
