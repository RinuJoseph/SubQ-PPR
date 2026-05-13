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



## QA (F1 / EM, top-5 passages)

| Dataset | Method | F1 | 
|---|---|---|
| **MuSiQue** | HippoRAG-2 (paper) | 0.4809 |
|  | **SubQ-PPR (ours)** | 0.477| 
| **2WikiMultiHopQA** | HippoRAG-2 (paper) | 0.7060 |
|  | **SubQ-PPR (ours)** | **0.657** |
|  | Δ | −4.88 pt | −3.80 pt |
| **HotpotQA** | HippoRAG-2 (paper) | 0.7130 | 
|  | **SubQ-PPR (ours)** | 0.747| 
| **NQ_rear** | HippoRAG-2 (paper) | not published | not published |
|  | **SubQ-PPR (ours)** | 0.502 | 



---
