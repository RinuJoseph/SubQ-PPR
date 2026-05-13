"""Prompt templates used by SQ-PPR.

Each prompt is in its own module:
  - ner.py                  — NER prompt (verbatim from HippoRAG-2)
  - triple_extraction.py    — OpenIE triple extraction (verbatim from HippoRAG-2)
  - rag_qa_musique.py       — final QA prompt (verbatim from HippoRAG-2)
  - fact_filter.py          — gpt-4o-mini fact filter (ours)
  - query_decomposition.py  — Claude / GPT decomposition prompt
"""
from . import ner, triple_extraction, rag_qa_musique, fact_filter, query_decomposition

__all__ = ["ner", "triple_extraction", "rag_qa_musique",
            "fact_filter", "query_decomposition"]
