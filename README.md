# SubQ-PPR — Sub-Question-Aware Personalized PageRank for Multi-Hop Retrieval

A compact reimplementation of HippoRAG-2 that does Personalised PageRank on a
**per-query subgraph** seeded by LLM-generated sub-questions, instead of the
full-corpus graph the original paper uses.

The change buys two things:

1. **Cost** — every stage runs on `gpt-4o-mini`. Index-time NER + OpenIE,
   query-time fact filter, QA reading — all on the cheapest model. No
   Llama-3.3-70B + DSPy.
2. **Match-or-beat HippoRAG-2 on retrieval recall**, with the full pipeline
   costing ~$0.50 per 1 000 MuSiQue queries.

The repo reproduces HippoRAG-2's published numbers across the multi-hop
benchmarks they evaluate on (MuSiQue, HotpotQA, 2WikiMultiHopQA, HoVer,
PopQA), then re-runs each with the SubQ-PPR pipeline on top.

See [`results.md`](results.md) for the side-by-side numbers.

---

## The idea

HippoRAG-2 builds **one big knowledge graph over the entire corpus** at index
time, then runs Personalised PageRank from "entity seeds" produced by a
query→fact reranker. The graph stays the same for every query; only the
reset vector changes.

SubQ-PPR keeps the same graph topology rules (entities from OpenIE triple
subjects/objects, `hr2_norm` normalisation, synonym edges ≥ 0.80 cosine,
damping 0.5, passage_node_weight 0.05, own-only scoring) — but **assembles
the graph per query** out of a small candidate pool. The pool is built by:

1. Dense retrieval of the **main question** → top-50 passages.
2. Dense retrieval of each LLM-generated **sub-question** → top-10 passages.
3. Union, deduplicate. Typical pool size: **~60 passages, ~770 entity
   nodes, ~1 600 edges**.

The intuition: most of the full HippoRAG-2 graph is irrelevant for any
given query. Sub-question retrieval is the part that brings in the
otherwise-unreachable hops (e.g. for *"What is the capital of the country
where X was born?"*, the second sub-question pulls passages about X's
birthplace that the main query's cosine alone would miss).

```
                            ┌─────────────────────────┐
   query                    │ Step 0 — decompose      │
                  ──────────│   gpt-4o-mini (cached)  │
                            │   → sub-questions       │
                            └────────┬────────────────┘
                                     │
                            ┌────────▼────────────────┐
   sub-q DPR ────────────► │ Step 1 — pool           │
                            │   mq top-50 ∪ sq top-10 │
                            │   deduped (~60 docs)    │
                            └────────┬────────────────┘
                                     │
                            ┌────────▼────────────────┐
   chunk triples ─────────► │ Step 2 — graph         │
   (cached NER + OpenIE)   │   entity nodes from     │
                            │   triple subjects/objects
                            │   hr2_norm, syn ≥ 0.80  │
                            └────────┬────────────────┘
                                     │
                            ┌────────▼────────────────┐
   fact embs (cached)       │ Step 3 — LLM filter    │
   gpt-4o-mini  ──────────► │   top-30 by cosine      │
                            │   → ~4.5 kept           │
                            └────────┬────────────────┘
                                     │
                            ┌────────▼────────────────┐
                            │ Step 4 — PPR            │
                            │   entity reset from kept│
                            │     facts (HippoRAG-2   │
                            │     algorithm)          │
                            │   passage reset =       │
                            │     mq_DPR × 0.05       │
                            │   damping = 0.5         │
                            └────────┬────────────────┘
                                     │
                                     ▼
                              top-K passages → QA
                              (gpt-4o-mini rag_qa_musique)
```

---

## Repository layout

```
SubQ-PPR/
├── README.md             ← this file
├── results.md            ← side-by-side numbers vs HippoRAG-2 paper
├── requirements.txt
├── configs/
│   ├── default.yaml      ← all hyperparameters; one file per dataset
│   └── mq100.yaml        ← alt config: mq_top=100 (analysis)
├── dataset/              ← drop the HippoRAG-2 dataset JSONs here
│                            (musique.json + musique_corpus.json, etc.)
├── prompts/              ← LLM prompts, one file each
│   ├── ner.py                  (verbatim from HippoRAG-2)
│   ├── triple_extraction.py    (verbatim from HippoRAG-2)
│   ├── rag_qa_musique.py       (verbatim from HippoRAG-2)
│   ├── fact_filter.py          (ours)
│   └── query_decomposition.py  (decompose_v1 prompt)
├── sqppr/                ← library
│   ├── datasets.py       ← per-dataset adapters (6 supported)
│   ├── data.py           ← uniform loader on top of adapters
│   ├── embedding.py      ← NV-Embed-v2 wrapper, GPU-lock + disk cache
│   ├── openie.py         ← NER + triple extraction (HippoRAG-2 class)
│   ├── decomposition.py  ← gpt-4o-mini query decomposition (cached)
│   ├── pool.py           ← mq50 ∪ sq10 pool builder
│   ├── graph.py          ← strict HippoRAG-2 per-query graph
│   ├── fact_filter.py    ← gpt-4o-mini fact filter, top-K / cap-out
│   ├── ppr.py            ← PPR reset construction + igraph PRPACK
│   ├── retrieval.py      ← end-to-end retrieve_one
│   ├── qa.py             ← QA on top-5 with rag_qa_musique
│   ├── metrics.py        ← recall_at_k, compute_f1, compute_em
│   ├── cache_prep.py     ← idempotent cache build
│   └── utils.py          ← config loader, hashing, JSON IO
└── scripts/              ← entry points
    ├── build_cache.py    ← decomp + passage embs + NER + triples
    ├── run_retrieval.py  ← pool → graph → filter → PPR
    ├── run_qa.py         ← QA on saved top-5
    ├── eval.py           ← aggregate report + HippoRAG-2 deltas
    └── run_all_datasets.py
```

---

## Setup

### Prerequisites

- Python ≥ 3.10
- A CUDA GPU (any size — NV-Embed-v2 fits in <10 GB)
- `OPENAI_API_KEY` exported in env (gpt-4o-mini is the only LLM used)
- HippoRAG-2's source code on PYTHONPATH — the encoder + OpenIE wrapper
  classes are imported from `src.hipporag`. Clone it next to this repo:

  ```bash
  git clone https://github.com/OSU-NLP-Group/HippoRAG.git
  # `sqppr/embedding.py` defaults to /workspace/storage/GraphRAG/HippoRAG;
  # edit the HIPPO_ROOT path there if your checkout is elsewhere.
  ```

### Install

```bash
pip install -r requirements.txt
```

### Datasets

This repo does **not** redistribute the HippoRAG-2 evaluation data. Download
the JSON dataset + corpus files from the HippoRAG-2 repo and place them
under `dataset/`:

```
dataset/
├── musique.json                  musique_corpus.json
├── hotpotqa.json                 hotpotqa_corpus.json
├── 2wikimultihopqa.json          2wikimultihopqa_corpus.json
├── hover.json                    hover_corpus.json
├── popqa.json                    popqa_corpus.json
└── nq_rear.json                  nq_rear_corpus.json
```

The adapter in `sqppr/datasets.py` handles the format differences between
them. All six load with 100% chunk-level gold resolution.

---

## How to reproduce

Each dataset's pipeline runs as four stages, all idempotent (existing cache
entries short-circuit).

### Single dataset

```bash
# Build the cache (decomp + passage embs + NER + triples for pool chunks)
python scripts/build_cache.py    --config configs/default.yaml --dataset musique

# Run retrieval (saves results/musique/retrieval/summary.{json,csv})
python scripts/run_retrieval.py  --config configs/default.yaml --dataset musique

# Run QA on the saved top-5 (saves results/musique/qa/qa_results.{json,csv})
python scripts/run_qa.py         --config configs/default.yaml --dataset musique

# Aggregate + write results/musique/eval_report.{json,md}
python scripts/eval.py           --config configs/default.yaml --dataset musique
```

### All datasets

```bash
python scripts/run_all_datasets.py                       # 6 retrieval datasets
python scripts/run_all_datasets.py --skip-qa             # retrieval only
python scripts/run_all_datasets.py --datasets musique 2wikimultihopqa
```

### Cost expectations (first-time cold runs)

| Stage | LLM calls | Cost (gpt-4o-mini) |
|---|---|---|
| Step 0 — decompositions (1 000 queries) | ~1 000 | ~$0.10 |
| Step 1 — passage embeddings | 0 (NV-Embed only) | $0 (GPU) |
| Steps 2-3 — NER + triples (pool chunks, ~10K) | ~10 000 | ~$5-8 |
| Retrieval — LLM fact filter | ~1 000 | ~$0.50 |
| QA | ~1 000 | ~$0.30 |
| **Total per dataset** | **~13 000** | **~$6-9** |

Re-runs with warm cache: only the retrieval + QA LLM calls fire → ~$0.80.

### Configurable knobs (in `configs/default.yaml`)

```yaml
pool:
  mq_top:  50      # main-query top-K passages
  sq_top:  10      # each sub-question top-K passages

fact_filter:
  input_top_k:     30      # candidate facts sent to LLM (top-K by cosine)
  output_cap_k:    null    # null = open (no cap); int = hard cap

ppr:
  damping:              0.5     # HippoRAG-2 default
  passage_node_weight:  0.05    # HippoRAG-2 default
  scoring_rule:         own_only

qa:
  top_k_passages:       5
```

`configs/mq100.yaml` mirrors `default.yaml` with `mq_top: 100` and a separate
output dir, used to show the pool-size sensitivity in `results.md`.

---

## Cache layout

Every cache entry is keyed by content hash, so re-runs are O(0):

```
cache/<dataset>/
├── decompositions/<qid>.json           gpt-4o-mini sub-questions per query
├── embeddings/passage/passage_embeddings.npy   NV-Embed (N, 4096)
├── embeddings/text/<sha1>.npy          per-text embeddings (queries, facts)
├── ner_doc/<chunk_id>.json             chunk-level NER (cached LLM)
└── triples/<chunk_id>.json             chunk-level OpenIE triples (cached LLM)
```

`chunk_id = "chunk-" + md5(title + "\n" + text)` — matches HippoRAG-2's own
scheme, so caches built by either side are bit-compatible.

Passage embeddings are auto-bootstrapped from HippoRAG-2's pre-built
`outputs/<dataset>/.../vdb_chunk.parquet` files when present, skipping the
GPU re-encode entirely.

---

## What's NOT in this repo

- **Cache directories** — re-generate via `build_cache.py`.
- **Results directories** — re-generate by running the pipeline.
- **Tests** — internal smoke tests excluded per repo policy.
- **Datasets** — see Setup above.

---

## License & attribution

Several prompts and the OpenIE / NER class are imported verbatim from
[HippoRAG-2 (OSU-NLP-Group/HippoRAG)](https://github.com/OSU-NLP-Group/HippoRAG)
and used under their license. The per-query graph construction, LLM-filter
prompt, sub-question pool design, dataset adapters, and all pipeline glue
code are original to this repo.
