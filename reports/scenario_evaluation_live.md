| Scenario | Query | Age / domain | PersonHit@5 | ProfileAlign@5 | Avg KG@5 | Avg profile@5 | Domains@10 | EMA/graph-scored@10 | Top result |
|---|---|---|---|---|---|---|---|---|---|
| Actor/director lookup | James Cameron | 20 / - | 4/5 | 5/5 | 0.667 | 0.980 | finance, movie, show | 10/0 | James Cameron's Story of Science Fiction |
| Singer/artist lookup | Taylor Swift | 20 / - | 5/5 | 5/5 | 0.667 | 0.630 | book, movie, music | 10/0 | The Tortured Poets Department |
| Profession-aware learning | business analytics and learning | 30 / - | -- | 5/5 | 0.650 | 1.000 | book, finance | 10/0 | Big Data Analytics |
| Family-safe profile | fun animation adventure | 8 / - | -- | 5/5 | 0.667 | 1.000 | movie | 10/0 | The Ice Age Adventures of Buck Wild |
| Cross-media semantic intent | space adventure with aliens | 20 / - | -- | 5/5 | 0.600 | 0.896 | movie, show | 10/0 | Aliens in the Attic |
| Health product discovery | joint pain supplements and mobility support | 30 / health | -- | 5/5 | 0.900 | 1.000 | health | 10/0 | Research Verified Joint Support - 100% Natural Glucosamine, MSM, Turmeric, Boswellia, BioPerine - Vegan - 90 Capsules -1 Month Supply |
| Industrial supply lookup | protective safety gloves for the lab | 30 / industry | -- | 5/5 | 0.767 | 0.694 | industrial | 10/1 | Blue Vinyl Disposable Gloves Large 50 Pack - Latex Free, Powder Free Medical Exam Gloves - Surgical, Home, Cleaning, and Food Gloves - 3 Mil Thickness |
| Finance literacy discovery | personal budgeting and bookkeeping | 30 / finance | -- | 5/5 | 0.700 | 0.938 | finance | 10/0 | Budgeting 101 |
| Director lookup | Christopher Nolan | 30 / - | 5/5 | 5/5 | 0.667 | 0.640 | movie | 10/3 | The Odyssey |
| Author lookup | Stephen King | 30 / - | 4/5 | 5/5 | 0.667 | 0.834 | book, show | 10/0 | Carrie |
| Band lookup | Coldplay | 20 / - | 4/5 | 5/5 | 0.333 | 0.502 | movie, music | 10/2 | Coldplay: A Head Full of Dreams |
| Fantasy cross-media intent | magic and dragons | 20 / - | -- | 5/5 | 1.000 | 0.958 | book, movie, show | 10/0 | Mighty Jack and the Goblin King |
| Romance cross-media intent | romantic drama about love | 30 / - | -- | 5/5 | 0.550 | 0.992 | movie, show | 10/0 | The Interest of Love |
| Action and superhero intent | action thriller with superheroes | 30 / - | -- | 5/5 | 0.650 | 0.970 | movie | 10/0 | Hero |
| Health vitamin discovery | vitamin D and immune support | 30 / health | -- | 5/5 | 1.000 | 0.938 | health | 10/1 | Optimum Nutrition Serious Mass Weight Gainer Protein Powder, Vitamin C, Zinc and Vitamin D for Immune Support, Banana, 6 Pound (Packaging May Vary) |
| Industrial tool lookup | torque wrench and fastener tools | 30 / industry | -- | 5/5 | 0.760 | 0.854 | industrial | 10/0 | M12-1.75 x 30MM Socket Head Cap Screws, Allen Socket Drive, Din 912, Stainless Steel 304, Full Thread, Bright Finish, Machine Thread, Quantity 10 by Fastenere |
| Finance investing discovery | retirement investing basics | 30 / finance | -- | 5/5 | 0.600 | 0.758 | finance | 10/0 | Seven steps to financial freedom in retirement |

Cross-domain rate (top 10 spans >= 2 content types): 8/17
Mean PersonHit@5 over person queries: 0.88
Mean ProfileAlign@5: 1.00
Mean KG@5: 0.697   Mean profile@5: 0.858

| Scenario | Relevant in catalogue | P@10 | R@10 | NDCG@10 | MAP@10 | MRR@10 | HR@10 |
|---|---|---|---|---|---|---|---|
| Actor/director lookup | 15 | 0.80 | 0.5333 | 0.716 | 0.617 | 0.500 | 1 |
| Singer/artist lookup | 37 | 0.80 | 0.2162 | 0.870 | 0.800 | 1.000 | 1 |
| Profession-aware learning | 550 | 0.90 | 0.0164 | 0.927 | 0.866 | 1.000 | 1 |
| Family-safe profile | 5251 | 1.00 | 0.0019 | 1.000 | 1.000 | 1.000 | 1 |
| Cross-media semantic intent | 1772 | 1.00 | 0.0056 | 1.000 | 1.000 | 1.000 | 1 |
| Health product discovery | 3877 | 1.00 | 0.0026 | 1.000 | 1.000 | 1.000 | 1 |
| Industrial supply lookup | 988 | 1.00 | 0.0101 | 1.000 | 1.000 | 1.000 | 1 |
| Finance literacy discovery | 941 | 1.00 | 0.0106 | 1.000 | 1.000 | 1.000 | 1 |
| Director lookup | 15 | 0.80 | 0.5333 | 0.848 | 0.733 | 1.000 | 1 |
| Author lookup | 72 | 0.90 | 0.1250 | 0.890 | 0.790 | 1.000 | 1 |
| Band lookup | 38 | 0.90 | 0.2368 | 0.780 | 0.707 | 0.500 | 1 |
| Fantasy cross-media intent | 1531 | 1.00 | 0.0065 | 1.000 | 1.000 | 1.000 | 1 |
| Romance cross-media intent | 6558 | 1.00 | 0.0015 | 1.000 | 1.000 | 1.000 | 1 |
| Action and superhero intent | 3347 | 1.00 | 0.0030 | 1.000 | 1.000 | 1.000 | 1 |
| Health vitamin discovery | 3392 | 1.00 | 0.0029 | 1.000 | 1.000 | 1.000 | 1 |
| Industrial tool lookup | 816 | 1.00 | 0.0123 | 1.000 | 1.000 | 1.000 | 1 |
| Finance investing discovery | 705 | 1.00 | 0.0142 | 1.000 | 1.000 | 1.000 | 1 |
| Mean | | 0.947 | 0.1019 | 0.943 | 0.913 | 0.941 | 1.000 |
