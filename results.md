# Results — SubQ-PPR vs. HippoRAG-2

Comparison of SubQ-PPR's reproduced numbers against the HippoRAG-2 paper.


## Retrieval (R@5)

| Dataset | Method | R@5 |
|---|---|---|
| **MuSiQue** | HippoRAG-2 (paper) | 0.7418 |
|  | **SubQ-PPR (ours)** | **0.7493** |
| **2WikiMultiHopQA** | HippoRAG-2 (paper) | 0.9010 |
|  | **SubQ-PPR (ours)** | 0.8515 |
| **HotpotQA** | HippoRAG-2 (paper) | 0.9280 |
|  | **SubQ-PPR (ours)** | **0.9620** |
| **NQ_rear** | HippoRAG-2 (paper) | not published |
|  | **SubQ-PPR (ours)** | 0.7353 |



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
