Machine: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, 12 logical CPUs, encoder on cuda; catalog 120,747 items; top_k=10

| Path / stage | n | mean ms | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| POST /recommend (all) | 700 | 371.27 | 363.15 | 455.28 | 507.18 |
| POST /recommend (anonymous) | 100 | 345.85 | 342.05 | 426.59 | 488.75 |
| POST /recommend (personalised) | 600 | 375.50 | 369.78 | 458.80 | 508.37 |
|   stage: creator match | 700 | 46.20 | 44.41 | 63.12 | 74.05 |
|   stage: lexical match | 700 | 198.91 | 190.76 | 265.66 | 307.17 |
|   stage: query encoding | 700 | 12.64 | 12.00 | 14.04 | 23.36 |
|   stage: Qdrant ANN search | 700 | 85.19 | 82.63 | 106.33 | 126.39 |
|   stage: MySQL preference rerank | 700 | 6.37 | 7.17 | 8.53 | 9.48 |
|   stage: LightGCN rerank | 700 | 0.07 | 0.07 | 0.13 | 0.16 |
|   stage: EMA rerank | 700 | 6.54 | 7.25 | 9.33 | 10.55 |
|   stage: KG/profile rerank | 700 | 13.11 | 13.10 | 22.29 | 28.15 |
| EMA profile update (per event) | 2000 | 0.05 | 0.05 | 0.06 | 0.08 |

| Concurrent workers | requests | req/s |
|---|---|---|
| 1 | 300 | 2.7 |
| 4 | 300 | 3.2 |
| 8 | 300 | 3.3 |
