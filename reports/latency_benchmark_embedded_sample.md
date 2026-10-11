Machine: AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD, 12 logical CPUs, encoder on cuda; catalog 120,747 items; top_k=10

| Path / stage | n | mean ms | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| POST /recommend (all) | 28 | 1835.60 | 1817.94 | 1946.44 | 2083.94 |
| POST /recommend (anonymous) | 4 | 1863.26 | 1792.74 | 2077.35 | 2116.23 |
| POST /recommend (personalised) | 24 | 1830.99 | 1826.28 | 1900.51 | 1954.65 |
|   stage: creator match | 28 | 44.50 | 41.56 | 51.79 | 56.12 |
|   stage: lexical match | 28 | 192.74 | 185.29 | 221.25 | 221.96 |
|   stage: query encoding | 28 | 13.40 | 13.26 | 15.29 | 15.74 |
|   stage: Qdrant ANN search | 28 | 1557.35 | 1533.77 | 1644.43 | 1794.12 |
|   stage: MySQL preference rerank | 28 | 6.44 | 7.53 | 8.06 | 8.11 |
|   stage: LightGCN rerank | 28 | 0.09 | 0.08 | 0.18 | 0.21 |
|   stage: EMA rerank | 28 | 6.35 | 7.26 | 8.70 | 9.07 |
|   stage: KG/profile rerank | 28 | 12.53 | 12.46 | 21.42 | 21.89 |
| EMA profile update (per event) | 2000 | 0.03 | 0.03 | 0.03 | 0.04 |

| Concurrent workers | requests | req/s |
|---|---|---|
| 1 | 8 | 0.6 |
