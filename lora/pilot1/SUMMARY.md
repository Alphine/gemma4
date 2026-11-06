# LoRA pilot1 (RFT on 53 self-harvested synthetic trajectories) — 2026-10-07

Base: gemma-4-31b-it QAT (verified bf16 copy) loaded as NF4 (bf16-nf4) on 1x H100. LoRA r32/alpha 64/lr 1e-4, 2 epochs,
grad-accum 4 (42 optimizer steps), max-len 16384, loss on assistant tokens only. Data: runs/pilot1-sft (90 examples from 53
trajectories; 83 train / 7 val examples, split by task). Timeout 75 min, retries 0.

- Result: done, adapter written by this attempt, 820 tensors, 489,848,063 B, max |lora_B| 0.0020 (non-trivial).
- Throughput ~670-690 tok/s, peak GPU memory 41.7 GB, host RSS 65.3 GiB of 80 GiB, wall 52.4 min.
- Cost (reserved = billed model): GPU $3.45 + CPU $0.33 + RAM $0.56 = $4.34.
- Train loss: first 5 steps mean 0.119 -> end of epoch 1 (steps 17-21) 0.089 -> epoch 2 mean 0.055 -> last 5 steps 0.046.
- Val loss (7 examples, held-out tasks): 0.0905 after epoch 1, 0.0909 after epoch 2.
  => the second epoch cut train loss by ~40% but val loss did not move: memorisation, no extra generalisation (val set is tiny,
  so only a weak signal). The first epoch alone probably captured most of the useful change.

Not yet known: whether the adapter helps at all on dev/held-out tasks. Needs the paired A/B in kernels/gemma4-lb-eval
(43 dev tasks; also the 144 synthetic held-out tasks), which also exercises vLLM 0.19.1 LoRA serving (unverified).

## Adapter published (2026-10-07)
Private Kaggle dataset `<kaggle-owner>/gemma4-rft-adapter`, arm dir `rft-pilot1-sft-full/` = edit-early-v4 (the policy the
trajectories were harvested with: prompt + skills/swe-tools) + `adapter: rft_lora` (r32, 489,841,632 B,
sha256 a3d7ba48...e7927). The first publish accidentally used edit-early-v2 as base (prompt/tool mismatch with the training
data); fixed via BASE_ARM = RFT_BASE_ARM default edit-early-v4 and re-versioned the same day. Other publish bug fixed:
files baked into the Modal image have mtime 1970, which made `kaggle --dir-mode zip` fail ("ZIP does not support timestamps
before 1980") -> stamp mtimes before publishing.

Next: paired A/B in kernels/gemma4-lb-eval after the Kaggle GPU quota resets (2026-10-10): arm `rft-pilot1-sft-full` vs the
existing edit-early-v4 result (13/40 + 2/3 on the same 43 tasks).

## Paired A/B result (Modal H100, lora-ab1, 2026-10-08): NULL
144 held-out synthetic tasks, base (edit-early-v4) vs base+adapter, same seed/order, one vLLM session with LoRA enabled,
adapter probe OK (vLLM 0.19.1 loads and applies it; effect small).
- Resolved: base 70/144 (48.6%), adapter 72/144 (50.0%); diff +1.4 pts, 95% bootstrap CI [-4.2, +6.9], exact McNemar p = 0.80
  (9 only-adapter vs 7 only-base). The first 48 pairs looked better (+6.2 pts, p = 0.375) and regressed to the mean.
- Timeouts 59 -> 57 (41% -> 40%), empty patches 12 -> 9, skill-call malformed 10.8% -> 12.6%, adapter makes fewer skill calls
  (198 vs 277). No infra errors. Cost $15.10 for 288 runs ($0.047 per run).
Conclusion: RFT on 53 self-harvested easy synthetic trajectories (r32, 2 epochs, val flat in epoch 2) does not measurably help;
effects above ~+7 pts are excluded, small effects are not. The dominant failure (5.5-min timeouts, 41%) did not move. Decision:
do NOT spend on the full harvest (~$46-60) for self-distillation. Total Modal for this experiment ~ $28 (credits-covered).
