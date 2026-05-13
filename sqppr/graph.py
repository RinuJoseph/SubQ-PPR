"""Per-query graph construction (strict HippoRAG-2 style).

Entity nodes = unique normalized subjects/objects across all triples in pool.
Entity normalization = text_processing (hr2_norm).

Edges:
  passage ↔ entity:   weight 1 if entity in passage's triple-entities
  entity  ↔ entity:   weight 1 per triple (subject↔object), accumulated
  entity  ↔ entity:   weight = cosine if cosine(embed(a), embed(b)) >= 0.80

NER is NOT used for graph entity construction (this matches HippoRAG-2's
official indexing — entities come from triples).
"""
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import igraph as ig

from .utils import text_processing
from .openie import load_chunk_triples
from .embedding import embed_texts


def collect_triples_and_entities(pool: list, triples_cache_dir: Path
                                    ) -> Tuple[list, dict, set]:
    """Returns (triples, chunk_triple_entities, all_entities).

    triples:                list of {subject, predicate, object, subject_n,
                                       object_n, from_chunk}
    chunk_triple_entities:  dict cid -> set of normalized entities in that
                              chunk's triples
    all_entities:           set of all normalized entities across the pool
    """
    triples = []
    chunk_triple_entities = defaultdict(set)
    all_entities = set()
    for cid in pool:
        for t in load_chunk_triples(cid, triples_cache_dir):
            if not isinstance(t, (list, tuple)) or len(t) < 3:
                continue
            s_n = text_processing(t[0])
            o_n = text_processing(t[2])
            if not s_n or not o_n:
                continue
            triples.append({
                "subject":    t[0], "predicate": t[1], "object": t[2],
                "subject_n":  s_n,   "object_n":  o_n,
                "from_chunk": cid,
            })
            chunk_triple_entities[cid].add(s_n)
            chunk_triple_entities[cid].add(o_n)
            all_entities.add(s_n)
            all_entities.add(o_n)
    return triples, chunk_triple_entities, all_entities


def build_graph(pool: list, triples: list,
                  chunk_triple_entities: dict, all_entities: set,
                  emb_cache_dir: Path,
                  synonym_threshold: float = 0.80
                  ) -> Tuple[ig.Graph, Dict[str, int], List[str]]:
    """Build the undirected weighted igraph.

    Returns (G, name_to_idx, names).
    """
    ent_keys = sorted(all_entities)
    ent_texts = ent_keys[:]   # use normalized text as embedding input

    names: List[str] = []
    name_to_idx: Dict[str, int] = {}
    def add(name):
        if name in name_to_idx: return
        name_to_idx[name] = len(names); names.append(name)
    for cid in pool: add(f"doc:{cid}")
    for k in ent_keys: add(f"ent:{k}")

    edge_w: Dict[Tuple[int, int], float] = defaultdict(float)

    # passage <-> entity (FROM TRIPLES, not NER)
    for cid, ents in chunk_triple_entities.items():
        for n in ents:
            if n not in all_entities: continue
            a, b = name_to_idx[f"doc:{cid}"], name_to_idx[f"ent:{n}"]
            edge_w[(min(a, b), max(a, b))] += 1.0

    # entity <-> entity (triple subject<->object)
    for t in triples:
        s_n, o_n = t["subject_n"], t["object_n"]
        if s_n in all_entities and o_n in all_entities and s_n != o_n:
            a = name_to_idx[f"ent:{s_n}"]
            b = name_to_idx[f"ent:{o_n}"]
            edge_w[(min(a, b), max(a, b))] += 1.0

    # entity <-> entity (synonym cosine >= threshold)
    if len(ent_keys) >= 2:
        embs = embed_texts(ent_texts, emb_cache_dir, "query_to_passage")
        sim = embs @ embs.T
        iu, ju = np.triu_indices(len(ent_keys), k=1)
        mask = sim[iu, ju] >= synonym_threshold
        for i, j, c in zip(iu[mask], ju[mask], sim[iu, ju][mask]):
            ai = name_to_idx[f"ent:{ent_keys[i]}"]
            bi = name_to_idx[f"ent:{ent_keys[j]}"]
            edge_w[(min(ai, bi), max(ai, bi))] += float(c)

    G = ig.Graph(n=len(names), directed=False)
    G.vs["name"] = names
    if edge_w:
        edges = list(edge_w.keys())
        G.add_edges(edges)
        G.es["weight"] = [edge_w[e] for e in edges]
    return G, name_to_idx, names
