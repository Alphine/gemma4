| operator | class | candidates | verified bug | survived (ok) | crash | shape-mismatch | timeout | flaky | kill rate (verified/cand) | mean s/run |
|---|---|---|---|---|---|---|---|---|---|---|
| IDX_SWAP | indexing | 51 | 46 | 5 | 0 | 0 | 0 | 0 | 0.90 | 1.2 |
| IDX_OFF | indexing | 76 | 72 | 4 | 0 | 0 | 0 | 0 | 0.95 | 1.2 |
| IDX_DROP | indexing | 77 | 72 | 5 | 0 | 0 | 0 | 0 | 0.94 | 1.2 |
| BLOCK_SHAPE | blocking | 122 | 2 | 3 | 117 | 0 | 0 | 0 | 0.02 | 0.2 |
| GRID_OFF1 | grid | 29 | 29 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.2 |
| GRID_CDIV | grid | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.0 |
| GRID_PERM | grid | 7 | 6 | 1 | 0 | 0 | 0 | 0 | 0.86 | 1.3 |
| PROGID_AXIS | grid | 8 | 7 | 1 | 0 | 0 | 0 | 0 | 0.88 | 1.8 |
| WHEN_COND | accumulator | 9 | 8 | 1 | 0 | 0 | 0 | 0 | 0.89 | 1.7 |
| WHEN_DROP | accumulator | 10 | 7 | 3 | 0 | 0 | 0 | 0 | 0.70 | 1.7 |
| ACC_INIT | accumulator | 10 | 7 | 3 | 0 | 0 | 0 | 0 | 0.70 | 2.1 |
| REDUCE_AXIS | reduction | 15 | 12 | 1 | 2 | 0 | 0 | 0 | 0.80 | 1.3 |
| KEEPDIMS | reduction | 12 | 0 | 2 | 10 | 0 | 0 | 0 | 0.00 | 0.2 |
| DTYPE_ACC | precision | 7 | 6 | 0 | 1 | 0 | 0 | 0 | 0.86 | 1.0 |
| MASK_DROP | masking | 7 | 5 | 2 | 0 | 0 | 0 | 0 | 0.71 | 1.8 |
| CMP_FLIP | masking | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.8 |
| DS_OFF | slicing | 5 | 3 | 0 | 2 | 0 | 0 | 0 | 0.60 | 1.9 |
| LOOP_BOUND | slicing | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1.00 | 1.6 |
| CONST_DROP | scaling | 4 | 2 | 2 | 0 | 0 | 0 | 0 | 0.50 | 2.0 |
| SEM_PARALLEL | hw_semantics | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0.00 | 1.7 |
| **all** | | 463 | 296 | 35 | 132 | 0 | 0 | 0 | 0.64 | 1.0 |
