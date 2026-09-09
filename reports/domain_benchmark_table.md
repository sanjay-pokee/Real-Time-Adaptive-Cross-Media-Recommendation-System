| Domain | Dataset | Users | Items | Interactions | Recall@20 (Pop.) | Recall@20 (LightGCN) | NDCG@20 (LightGCN) | MRR@20 (LightGCN) | Lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Entertainment | Movies & TV | 50,000 | 52,263 | 914,346 | 0.0182 | 0.0288 | 0.0125 | 0.0080 | 1.59x |
| Industry | Industrial & Scientific | 33,959 | 19,024 | 238,752 | 0.0264 | 0.0326 | 0.0138 | 0.0087 | 1.24x |
| Health | Health & Household | 50,000 | 42,640 | 809,212 | 0.0221 | 0.0121 | 0.0046 | 0.0025 | 0.55x |

_Industrial & Scientific is an order of magnitude smaller (413k raw interactions vs 7.4M) and does not survive a 10-core filter, so it uses the source file's 5-core floor; the other two domains use 10-core._
