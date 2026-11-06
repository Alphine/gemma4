# Captions for the three writeup figures (each <= 45 words)

**fig_time_anatomy.png.** Mean wall-clock per run by arm (n=14/13/13 dev runs; 1,009 completed calls in panel b); whiskers: bootstrap 95% CI of mean wall. Tool execution is 0.7%; reasoning decode 60%. Source: A/B session-1 traces via research/top10/anatomy/build_runs.py, datasets/gemma4-lb-eval-ckpt/results_*.jsonl.

**fig_prose_vs_mech.png.** Reads+searches before first edit, 13 matched tasks (13 triples, single sample per cell); diamonds: mean with bootstrap 95% CI; hollow points never edited. Panel b: counts over 13 runs. Source: datasets/gemma4-lb-eval-ckpt/results_edit-early-{v6-locate,v7-decide-nt,v7-lean-nt}.jsonl.

**fig_noise_power.png.** P(best of N resubmissions of one config scores >= 12/58) versus its true score, two simulation models (A solid, B dashed; 1,500 sets per point). Rugs: our 11 scores and third-party replicates 7, 6, 4; band: 95% interval. Source: research/top10/strategy/s3_hit.json, s2_posterior.json, research/submissions_history.csv.
