# C-tpu v2: native TPU v5e-8 run of the Pallas mutant suite (2026-10-17)

Kernel `<kaggle-user>/gemma4-app-c-tpu-tpu` (version 2, libtpu refreshed, `--accelerator TpuV5E8`) COMPLETED. Output in `tpu-out-v2/` (results.json, variants.jsonl, smoke.jsonl, log).
The v1 failure (native Pallas blocked by old libtpu in the Kaggle TPU image) is resolved: the smoke test `add.base` ran natively on 8 TPU v5 lite devices.

## Result (V, 51 variants, interpret=True on CPU vs native TPU)
- Status: both ran 45, interpret ran / native error 5, both error 1. Verdict agreement (PASS/FAIL identical) 43 of 51; the 8 disagreements are these 6 mutants plus the 2 base kernels below.
- 6 mutants PASS in interpret mode but are caught on real hardware: 5 raise an error (below) and 1 (`matmul.k_outer`, GRID-ORDER, grid axis order) returns a wrong result natively (rel_err 1.06; correction added 2026-10-17, the first summary missed it). The 5 errors are:
  - BLK-TILING (add.blk_lane100, softmax.blk_4rows): "Pallas TPU lowering requires the last two dimensions ..." (block shape not tiling-aligned)
  - BLK-VMEM (add.huge_block_vmem): RESOURCE_EXHAUSTED CompileTimeScopedVmemOom
  - GRID-OOB (add.grid_big_oob): core halted unexpectedly
  - IDX-OOB (layernorm.g_map_i_oob): program continuator halted unexpectedly
- Two further disagreements are base kernels (matmul.base, attn.base): interpret PASS, native ran with rel_err 0.0023 and 0.0030 (TPU default matmul precision) and
  was marked FAIL by the suite tolerance. This is a tolerance artefact, not a kernel bug; the suite tolerance should be loosened for native matmul-class kernels.
- Remaining interpret-FAIL / native-FAIL cells (ARITH, IDX, INIT, MASK, PID, RED, ...) agree: these mutants are detected on both backends.

## Meaning for the paper / application track
The central claim, that CPU interpret mode misses hardware-only bug classes (tiling, VMEM, OOB), is now confirmed on real TPU hardware for 5 mutants of 4 classes (erroring)
(previously "interpret-mode only, native unverified"; with matmul.k_outer, 6 mutants of 5 classes). The suite needs one fix before release: native tolerance for matmul/attn base kernels.
