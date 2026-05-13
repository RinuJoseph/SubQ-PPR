"""OpenIE triple-extraction prompt — copied verbatim from HippoRAG-2
(src/hipporag/prompts/templates/triple_extraction.py).

Conditioned on the NER list per passage. Returns a JSON list of triples.
"""
from .ner import one_shot_ner_paragraph, one_shot_ner_output


ner_conditioned_re_system = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""


ner_conditioned_re_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""


ner_conditioned_re_input = ner_conditioned_re_frame.format(
    passage=one_shot_ner_paragraph,
    named_entity_json=one_shot_ner_output,
)


ner_conditioned_re_output = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"],
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""


prompt_template = [
    {"role": "system",    "content": ner_conditioned_re_system},
    {"role": "user",      "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    # placeholder filled in at call time with {passage} + {named_entity_json}
    {"role": "user",      "content": ner_conditioned_re_frame},
]
