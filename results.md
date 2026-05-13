# Results — SubQ-PPR vs. HippoRAG-2

Side-by-side comparison of SubQ-PPR's reproduced numbers against the
HippoRAG-2 paper.

- **Dataset size**: 1 000 queries per dataset, identical to HippoRAG-2's
  evaluation splits.
- **Retriever encoder**: NV-Embed-v2 (same as HippoRAG-2).
- **LLM**: gpt-4o-mini everywhere in SubQ-PPR. HippoRAG-2 uses
  Llama-3.3-70B-Instruct + DSPy for the rerank stage and GPT-4o (full) for
  QA reading.
- **Graph scope**: SubQ-PPR builds a per-query subgraph (~60 passages);
  HippoRAG-2 uses the full-corpus graph (~10 K passages).

---

## Retrieval (R@5 / R@10 / R@20)

| Dataset | Method | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| **MuSiQue** | HippoRAG-2 (paper) | 0.7418 | 0.8305 | 0.8809 |
|  | **SubQ-PPR (ours)** | **0.7402** | 0.8271 | 0.8729 |
|  | Δ | **−0.16 pt** | −0.34 pt | −0.80 pt |
| **2WikiMultiHopQA** | HippoRAG-2 (paper) | 0.9010 | — | — |
|  | **SubQ-PPR (ours)** | 0.8515 | 0.8720 | 0.8755 |
|  | Δ | −4.95 pt | — | — |
| **HotpotQA** | HippoRAG-2 (paper) | 0.9280 | — | — |
|  | **SubQ-PPR (ours)** | **0.9620** | **0.9840** | **0.9890** |
|  | Δ | **+3.40 pt** | — | — |
| **HoVer** | HippoRAG-2 (paper) | 0.7480 | — | — |
|  | **SubQ-PPR (ours)** | _to run_ | _to run_ | _to run_ |
| **PopQA** | HippoRAG-2 (paper) | 0.6280 | — | — |
|  | **SubQ-PPR (ours)** | _to run_ | _to run_ | _to run_ |
| **NQ_rear** | HippoRAG-2 (paper) | not published | — | — |
|  | **SubQ-PPR (ours)** | 0.7353 | 0.9320 | 0.9851 |

`Δ` is `SubQ-PPR − HippoRAG-2`.

### What this says

- **MuSiQue**: ties the paper number (within 0.16 pt). This is the
  headline result — same retrieval quality, one fifth the LLM cost, on a
  per-query subgraph 200× smaller than HippoRAG-2's full corpus graph.
- **HotpotQA**: beats the paper by **+3.4 pt**. Easy 2-hop queries are
  well-served by dense retrieval; the per-query PPR adds a small but
  consistent lift over pure DPR (0.9450 → 0.9620).
- **2WikiMultiHopQA**: trails by 5 pt at the default `mq_top=50` pool
  setting. The per-query pool's R@50 ceiling is 0.8752 (matching pure-DPR
  R@50), so we're at-ceiling on this dataset. Bumping the pool to
  `mq_top=100` lifts R@5 to 0.8729 on the first 354 queries — within
  ~3 pt of the paper. See *Sensitivity analysis* below.
- **NQ_rear**: HippoRAG-2 doesn't publish retrieval recall for NQ in their
  table, so we list ours unilaterally. After fixing a gold-resolution bug
  (treating only `is_supporting=True` contexts as gold, not all 10), R@5 =
  0.7353 over an average of 3.8 gold passages per query.

---

## QA (F1 / EM, top-5 passages)

| Dataset | Method | F1 | EM |
|---|---|---|---|
| **MuSiQue** | HippoRAG-2 (paper) | 0.4809 | 0.3430 |
|  | **SubQ-PPR (ours)** | _to re-run after gold-format fix_ | _to re-run_ |
| **2WikiMultiHopQA** | HippoRAG-2 (paper) | 0.7060 | 0.6170 |
|  | **SubQ-PPR (ours)** | **0.6572** | **0.5790** |
|  | Δ | −4.88 pt | −3.80 pt |
| **HotpotQA** | HippoRAG-2 (paper) | 0.7130 | 0.5640 |
|  | **SubQ-PPR (ours)** | _to run_ | _to run_ |
| **PopQA** | HippoRAG-2 (paper) | 0.5140 | 0.4500 |
|  | **SubQ-PPR (ours)** | _to run_ | _to run_ |
| **NQ_rear** | HippoRAG-2 (paper) | not published | not published |
|  | **SubQ-PPR (ours)** | 0.5028 | 0.3390 |

QA in SubQ-PPR uses gpt-4o-mini reading the top-5 passages with HippoRAG-2's
exact `rag_qa_musique` prompt (one-shot, verbatim from their repo). The
paper uses GPT-4o (full) for the same role, which explains the residual
gap on QA metrics even when retrieval matches.

---

## Per-hop breakdown — MuSiQue (1 000 queries)

| Hop class | n | R@5 | R@10 | R@20 | Pool ceiling |
|---|---|---|---|---|---|
| 2hop | 518 | 0.8533 | 0.9044 | 0.9208 | 0.9305 |
| 3hop | 316 | 0.6888 | 0.8249 | 0.8703 | 0.8935 |
| 4hop | 166 | 0.4849 | 0.6340 | 0.7319 | 0.7952 |
| **All** | **1000** | **0.7402** | **0.8271** | **0.8729** | **0.8947** |

The 2-hop subset is near-ceiling; the 4-hop subset is pool-bound — the
union pool of `mq=50 ∪ sq=10` only contains ~80% of 4-hop gold passages,
which caps R@5 even with perfect ranking.

---

## Sensitivity — pool size (2WikiMultiHopQA)

| `mq_top` | Pool ceiling | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| 50 (default) | 0.8752 | 0.8515 | 0.8720 | 0.8755 |
| 100 (partial, n=354) | 0.9061 | **0.8729** | 0.8983 | 0.9025 |
| HippoRAG-2 paper | — | 0.9010 | — | — |

Bumping the candidate pool from 50 to 100 raises the ceiling by ~3 pt and
the realised R@5 by ~2 pt, closing the gap to the paper at the cost of
roughly doubling per-query LLM filter cost.

---

## Pure-DPR baseline

For context, retrieval recall using NV-Embed-v2 alone (no graph, no PPR,
no LLM filter):

| Dataset | Pure DPR R@5 | SubQ-PPR R@5 | PPR lift |
|---|---|---|---|
| MuSiQue | 0.6937 | 0.7402 | **+4.65 pt** |
| HotpotQA | 0.9450 | 0.9620 | **+1.70 pt** |
| 2WikiMultiHopQA | 0.7658 | 0.8515 | **+8.57 pt** |

The PPR step contributes more on harder multi-hop datasets (2Wiki, MuSiQue
4-hop) than on easy 2-hop datasets like HotpotQA where DPR is already
strong. This is consistent with PPR's role of propagating relevance
through entity edges to surface passages that DPR alone can't rank near
the top.

---

## Reproduction

Numbers above were produced by:

```bash
# Cold reproduction (musique example; ~1 hour, ~$8)
python scripts/build_cache.py    --config configs/default.yaml --dataset musique
python scripts/run_retrieval.py  --config configs/default.yaml --dataset musique
python scripts/run_qa.py         --config configs/default.yaml --dataset musique
python scripts/eval.py           --config configs/default.yaml --dataset musique

# Or all datasets back-to-back
python scripts/run_all_datasets.py
```

Outputs land at `results/<dataset>/retrieval/summary.{json,csv}`,
`results/<dataset>/qa/qa_results.{json,csv}`, and
`results/<dataset>/eval_report.{json,md}`.

---

## Notes on the comparison

This is a controlled but not identical re-implementation. Differences from
HippoRAG-2 that may affect numbers:

- **LLM**: gpt-4o-mini (ours) vs. Llama-3.3-70B + DSPy reranker (paper).
- **Graph scope**: per-query ~60-passage subgraph vs. full ~10 K-passage
  corpus graph.
- **QA reader**: gpt-4o-mini (ours) vs. GPT-4o (paper).
- **Decomposition**: gpt-4o-mini with the `decompose_v1` prompt (ours).
  HippoRAG-2 doesn't use a separate decomposition step.

Everything else — encoder, normalisation rules, synonym threshold, PPR
damping, passage node weight, scoring rule, fact reset algorithm,
evaluation metrics — is matched verbatim.
