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
| **NQ_rear** | HippoRAG-2 (paper) | 0.764 |
|  | **SubQ-PPR (ours)** | 0.7353 |



## QA Results (F1, top-5 passages)

| Dataset | Method | F1 |
|---|---|---:|
| **MuSiQue** | HippoRAG-2 (paper) | 0.4809 |
|  | **SubQ-PPR (ours)** | 0.4770 |
| **2WikiMultiHopQA** | HippoRAG-2 (paper) | 0.7060 |
|  | **SubQ-PPR (ours)** | 0.6570 |
| **HotpotQA** | HippoRAG-2 (paper) | 0.7130 |
|  | **SubQ-PPR (ours)** | **0.7470** |
|  | Δ | **+3.40 pt** |
| **NQ_rear** | HippoRAG-2 (paper) | 0.5020 |
|  | **SubQ-PPR (ours)** | 0.5020 |



---
