# Results — SubQ-PPR vs. HippoRAG-2

Comparison of SubQ-PPR's reproduced numbers against the HippoRAG-2 paper.


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
| **PopQA** | HippoRAG-2 (paper) | 0.6280 | — | — |
|  | **SubQ-PPR (ours)** | _to run_ | _to run_ | _to run_ |
| **NQ_rear** | HippoRAG-2 (paper) | not published | — | — |
|  | **SubQ-PPR (ours)** | 0.7353 | 0.9320 | 0.9851 |

`Δ` is `SubQ-PPR − HippoRAG-2`.



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
| **NQ_rear** | HippoRAG-2 (paper) | not published | not published |
|  | **SubQ-PPR (ours)** | 0.5028 | 0.3390 |



---
