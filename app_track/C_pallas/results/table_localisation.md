| method | n | acc@1 | acc@3 | acc@5 | acc@10 | MRR | median rank |
|---|---|---|---|---|---|---|---|
| B0_random | 296 | 0.024 | 0.111 | 0.155 | 0.314 | 0.121 | 17 |
| B1_pallas_token_prior | 296 | 0.193 | 0.622 | 0.814 | 0.936 | 0.444 | 3 |
| B2a_catalogue_only | 296 | 0.132 | 0.490 | 0.726 | 0.929 | 0.369 | 4 |
| B2b_catalogue+diag_classifier | 296 | 0.382 | 0.784 | 0.899 | 0.956 | 0.596 | 2 |
| B3_B2b+graph_propagation | 296 | 0.351 | 0.757 | 0.875 | 0.939 | 0.575 | 2 |

mean candidate lines per task: 63.5; diagnostic->bug-class accuracy (leave-one-seed-out RF): 0.7567567567567568 vs majority 0.642
