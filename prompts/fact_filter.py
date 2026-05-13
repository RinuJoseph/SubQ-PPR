"""LLM fact filter prompt — ours.

Given a multi-hop question, its sub-questions, and 30 candidate OpenIE facts
ranked by cosine, ask gpt-4o-mini to identify which facts contain evidence
needed to answer the main question or any sub-question.

Output format: comma-separated 1-based indices, or 'NONE'.

Used by src/fact_filter.py.
"""

FACT_FILTER_SYSTEM = (
    "You are an expert at filtering knowledge-graph facts for multi-hop QA. "
    "Identify facts that contain evidence needed to answer the main question "
    "OR any of its sub-questions."
)


def build_user_message(main_query: str, sub_questions: list, candidate_facts: list) -> str:
    """Render the user message for the fact-filter call.

    candidate_facts: list of (triple_idx:int, cosine_score:float,
                              (subject:str, predicate:str, object:str))
    Returns the user-content string.
    """
    sub_block = (
        "\n".join(f"  - {sq}" for sq in sub_questions)
        if sub_questions else "  (none)"
    )
    facts_block = "\n".join(
        f"[{i+1}] ({s!r}, {p!r}, {o!r})"
        for i, (_, _, (s, p, o)) in enumerate(candidate_facts)
    )
    return (
        f"Main question:\n  {main_query}\n\n"
        f"Sub-questions (intermediate hops):\n{sub_block}\n\n"
        f"Candidate facts (numbered 1..{len(candidate_facts)}):\n"
        f"{facts_block}\n\n"
        f"Output the numbers of facts that contain evidence for the "
        f"main question or any sub-question. Comma-separated list "
        f"(e.g. '3, 7, 12') or 'NONE'. No prose."
    )


def build_messages(main_query: str, sub_questions: list, candidate_facts: list) -> list:
    return [
        {"role": "system", "content": FACT_FILTER_SYSTEM},
        {"role": "user",   "content": build_user_message(main_query, sub_questions, candidate_facts)},
    ]
