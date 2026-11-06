> SUPERSEDED (2026-10-17): this is the history of the first TPU attempt, which never started. The native run completed later; see README.md and RESULT_v2.md. One correction to RESULT_v2.md: matmul.k_outer is a sixth interpret-PASS / native-FAIL mutant (wrong result, rel err 1.06).

# C-tpu: is the interpret-defined Pallas bug benchmark valid on real TPU? (2026-10-10)

Status: UNVERIFIED. The native-TPU job was accepted but stayed QUEUED for more than 1 h 50 min and never started; no native numbers exist, so the interpret-vs-TPU agreement is NOT measured. The path must not be claimed as validated in the paper.

## What was established
- Accelerator: `TpuV5E8` (v5e-8), pushed with `python -m kaggle kernels push -p <dir> -t 1800 --accelerator TpuV5E8` (metadata `enable_tpu: true`, `machine_shape: TpuV5E8`, `enable_internet: true`). A script kernel can get a TPU via the CLI. `Tpu1VmV38` (v3-8) was not tried.
- Quota reported by the platform: TPU 20.00 h/week, 0.00 h used, refresh 2026-10-17. One TPU push was used; one session remains.
- TPU image jax/jaxlib/libtpu versions: unknown (Kaggle TPU docs page could not be extracted). CPU image: jax/jaxlib 0.11.1, Python 3.13.
- Kernel `<kaggle-user>/gemma4-app-c-tpu-tpu` is still queued and cannot be cancelled via the CLI; it runs when a TPU frees up.

## Measured so far: CPU interpret mode (jax 0.11.1, kernel run 29 s)
- 51 variants: 6 baselines + 45 mutants over 8 operators (IDX, BLK, MASK, INIT, RED, GRID, GUARD, PID) and an ARITH group. Oracle: max|out-ref|/max|ref| < 2e-4 against a float64 numpy reference, fixed seeds.
- All 6 baselines pass. 38 of 45 mutants fail in interpret mode (84%).
- 7 mutants PASS interpret mode, so an interpret-defined benchmark would label them "not a bug":
  - `layernorm.no_eps`: genuinely near-equivalent.
  - `add.blk_lane100`: lane block 100 not a multiple of 128.
  - `softmax.blk_4rows`: sublane block 4 not a multiple of 8.
  - `add.huge_block_vmem`: 3 x 32 MB blocks, far above VMEM.
  - `add.grid_big_oob`: grid runs one block past the array.
  - `layernorm.g_map_i_oob`: index_map past a size-1 array.
  - `matmul.k_outer`: k outermost grid axis, output blocks revisited non-consecutively.
- Interpret-mode semantics observed: out-of-bounds reads in ragged blocks return NaN; an input index_map past the array gives a wrong result; an out-of-bounds output block is silently dropped; a size-1 block indexed at i>0 is silently clamped; shape mismatches are caught at trace time.

## Hypotheses for hardware-only classes (unmeasured, from Pallas/Mosaic docs)
| class | variants | expected native result |
|---|---|---|
| (8,128) tiling | blk_lane100, blk_4rows | compile error |
| VMEM overflow | huge_block_vmem | compile/resource error |
| OOB block index | grid_big_oob, g_map_i_oob, x_off_oob | wrong result, DMA error or hang |
| non-consecutive output revisit | k_outer | probably FAIL |
| f32 matmul precision | matmul/attention baselines | may differ at 2e-4 |

If confirmed, about 6 of 45 mutants (13%) would be mislabelled as passing by an interpret-defined ground truth. Implication for any write-up: task statements must say "numerics bugs observable under a sequential CPU interpreter", and hardware-only classes must be listed separately.

## Limits
Small suite (6 families, fixed small shapes, f32 only, one seed); no megacore, `dimension_semantics`, bf16 tiling or multi-chip effects; one tolerance choice (raw errors stored); queue wait unbounded.

## Collect the TPU result when it runs
```
python -m kaggle kernels status <kaggle-user>/gemma4-app-c-tpu-tpu
python -m kaggle kernels output <kaggle-user>/gemma4-app-c-tpu-tpu -p research/app-track/C-tpu/tpu-out
python research/app-track/C-tpu/summarize.py research/app-track/C-tpu/tpu-out/results.json
```
Files: `pallas_tpu_suite.py`, `summarize.py`, `cpu-out/`, kernels `gemma4-app-c-tpu-cpu` (done), `gemma4-app-c-tpu-tpu` (queued).
