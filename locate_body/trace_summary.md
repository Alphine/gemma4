Runs that called locate.py (ab1 base arm + pilot1 harvest, synthetic tasks): n=153 (153 tasks).

| quantity | value (95%% CI over runs) |
|---|---|
| printed code holds ALL gold lines (agent could edit without a read) | 28.1 [20.9, 35.3] |
| printed code holds ANY gold line | 36.6 [28.8, 44.4] |
| printed code does NOT hold the gold lines (wrong function/file) | 63.4 [55.6, 71.2] |
| the agent's first read after locate lies >=70% inside the printed code | 17.6 [11.8, 23.5] |
| the agent's first read after locate touches a gold line (v6-like behaviour: first read already right) | 30.7 [23.5, 38.6] |
| first read hits gold AND is inside the printed code (a clean saved read) | 7.2 [3.3, 11.8] |
| first read is inside the printed code but misses gold (agent reads this wrong code today as well) | 10.5 [5.9, 15.7] |
| runs with a read before the first source edit | 87.6 [82.4, 92.8] |

Saved agent turns (reads before the first source edit whose lines are >=70%% inside the printed code), per run:

| | mean per run (95%% CI) |
|---|---|
| saved LLM calls | 0.48 [0.35, 0.62] |
| saved seconds, lower bound (only the call itself: 0.44 s + 0.0221 s x 55 tokens per saved turn) | 0.8 [0.6, 1.0] |
| saved seconds, upper bound (whole turn incl. its reasoning tokens: 135 tokens per saved turn on average) | 1.7 [1.0, 2.4] |
| share of runs with >=1 saved call | 31.4 [24.2, 38.6] |

Context: mean completion tokens per run 8171 (= 181 s of generation at 0.0221 s/token + 16 s call overhead at 36 calls).

Added prompt tokens per run (body chars / 3.3): mean 485, p90 720. Compaction trigger 14336 prompt tokens: runs whose prompt ever exceeds it 108 -> 111 of 153 (steps above it: 206 -> 413 of 5449).

| style | n | all gold lines printed | saved calls/run |
|---|---|---|---|
| observed | 61 | 31.1 [19.7, 42.6] | 0.38 |
| repro | 25 | 40.0 [20.0, 60.0] | 0.80 |
| terse | 67 | 20.9 [11.9, 31.3] | 0.46 |

In these runs: timeout share 31%, resolved 52%, median agent minutes 4.45 (budget 5.5 min ab1 / 10 min pilot1).