"""Query decomposition prompt — verbatim copy of decompose_v1.

Source: Query-Decompose/prompts/decompose_v1.txt
"""

PROMPT_TEMPLATE = """\
You are an expert at decomposing complex multi-hop questions into a sequence
of simple, sub-questions that can be answered one at a time.

## Task
Given a complex question, break it down into the minimum number of ordered
sub-questions such that:
1. Each sub-question asks for ONE specific piece of information.
2. Each sub-question is self-contained and answerable independently (given
   the answers to previous sub-questions).
3. Later sub-questions may reference the answers of earlier ones using
   natural language placeholders like "the person from Q1" or "the answer
   to Q2".

## Rules
- Do NOT skip reasoning steps; each hop in the question becomes its own
  sub-question.
- Do NOT merge two facts into one sub-question.
- Keep sub-questions short, clear, and in natural language.
- Preserve the logical order: the output of one sub-question feeds into
  the next.
- Do not answer the sub-questions — only decompose.

## Output Format
Original Question: <restate the input question>

Sub-questions:
Q1: <first sub question>
Q2: <next question, referencing Q1's answer if needed>
Q3: <next question, referencing previous answers if needed>
...
Qn: <last sub-question>

## Examples

Example 1
Original Question: "In which year was the university attended by the author
of '1984' founded?"

Sub-questions:
Q1: Who is the author of the novel "1984"?
Q2: Which university did the answer to Q1 attend?
Q3: In what year was the answer to Q2 founded?

Example 2
Original Question: "What is the official language of the country that hosted
the 2016 Summer Olympics?"

Sub-questions:
Q1: Which country hosted the 2016 Summer Olympics?
Q2: What is the official language of the answer to Q1?

Example 3
Original Question: "Who directed the highest-grossing film of the actor who
played Iron Man in the Marvel Cinematic Universe?"

Sub-questions:
Q1: Which actor played Iron Man in the Marvel Cinematic Universe?
Q2: What is the highest-grossing film of the answer to Q1?
Q3: Who directed the answer to Q2?

## Now decompose the following question:
Original Question: "{question}\""""


def build_messages(question: str):
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]
