# Pallas mutant suite: CPU interpret mode vs native TPU v5e

A self-contained benchmark of 51 small Pallas kernels (6 baselines + 45 mutants) used to ask one question: do bugs that an
interpret-mode (CPU) harness labels "correct" also behave correctly on real TPU hardware? Answer on one TPU generation (v5e):
no, for 6 of the 45 mutants. Code is BSD-3-Clause; results are CC-BY-4.0.

## Files
| file | content |
|---|---|
| `pallas_tpu_suite.py` | the suite: 6 kernel families (add, matmul, softmax, layernorm, rowsum with ragged mask, causal attention), 51 variants, oracle = float64 numpy reference, fixed seeds. Parent process orchestrates one child process per batch of variants (one TPU client per child) and survives hangs/crashes. |
| `summarize.py` | agreement table and the list of hardware-only classes from a `results.json` |
| `cpu-out/` | result of the CPU-only run (jax 0.11.1, Python 3.13) |
| `tpu-out-v2/` | first native run (v5e-8, jax 0.11.0, libtpu 0.0.17): single native verdict at TPU default matmul precision |
| `tpu-out-v3/` | native run with the tolerance / precision fix described below |
| `REPORT.md` | history: the first TPU attempt (v1) never started; kept for the record |
| `RESULT_v2.md` | short write-up of the v2 run |

## How to run
CPU, interpret mode (no TPU needed; about 30 s):
```
pip install "jax==0.11.1" numpy
SUITE_OUT=out python pallas_tpu_suite.py          # writes out/results.json and out/variants.jsonl
python summarize.py out/results.json
```
Every variant is executed with `pallas_call(..., interpret=True)` on the CPU device. The `native_tpu` fields are `skipped`.

TPU (native Mosaic lowering): run the same command on a machine whose `jax.default_backend()` is `tpu`, e.g. a Kaggle script
kernel with `enable_tpu: true`, `machine_shape: TpuV5E8` and internet on. The Kaggle TPU image ships a libtpu that is too old for
the installed jax Pallas ("requires a libtpu version that's at most a month old"); the parent detects this with a smoke test
(`add.base`) and runs `pip install -U libtpu`, then points `TPU_LIBRARY_PATH` at the new `libtpu.so` (logged under
`env.libtpu_fix` in `results.json`). Disable with `SUITE_LIBTPU_FIX=0`. Each variant runs twice in one child: interpret on CPU,
then native on the TPU. Dot-based families (matmul, attn) additionally run natively at `jax.lax.Precision.HIGHEST`.
`SUITE_BUDGET_SEC` (default 1500) bounds the wall time.

## Result: native TPU confirms the interpret-mode gap (v5e, run v2)
51 variants; verdict = PASS / FAIL (rel. error against reference below tolerance) / ERR (raises). Interpret ran 50, errored 1
(`matmul.blk_k_mismatch`, a shape error caught at trace time, identical on both backends).

| outcome | count |
|---|---|
| interpret and native give the same verdict | 43 of 51 |
| interpret PASS, native ERR (hardware-only, 5 mutants) | 5 |
| interpret PASS, native FAIL (wrong result on hardware, 1 mutant) | 1 (`matmul.k_outer`) |
| interpret PASS, native FAIL on a *baseline* (tolerance artefact, fixed in v3) | 2 (`matmul.base`, `attn.base`) |

Mutants that pass in interpret mode but fail on hardware:

| mutant | class | interpret | native TPU v5e | native message (shortened) |
|---|---|---|---|---|
| `add.blk_lane100` | tiling: lane block 100, not a multiple of 128 | PASS (err 0) | ERR compile | "last two dimensions of your block shape are divisible by 8 and 128 respectively" |
| `softmax.blk_4rows` | tiling: sublane block 4, not a multiple of 8 | PASS | ERR compile | same |
| `add.huge_block_vmem` | VMEM: three 32 MB blocks | PASS | ERR compile | RESOURCE_EXHAUSTED CompileTimeScopedVmemOom |
| `add.grid_big_oob` | grid runs one block past the array | PASS | ERR runtime | "Core halted unexpectedly" |
| `layernorm.g_map_i_oob` | index_map past a size-1 array | PASS | ERR runtime | "program continuator has halted unexpectedly" |
| `matmul.k_outer` | reduction axis outermost, output block revisited non-consecutively | PASS | FAIL (rel err 1.06) | wrong result |

`matmul.k_outer` is a sixth hardware-only mutant that the first summary (`RESULT_v2.md`) counted among the "agreeing" cells by
mistake; it is listed above because interpret mode executes the grid sequentially and accumulates correctly, while on hardware
the output block is written back between non-consecutive visits. So 6 of 45 mutants (13%) are mislabelled "not a bug" by an
interpret-defined ground truth. The seventh interpret-PASS mutant, `layernorm.no_eps`, passes on both backends (near-equivalent,
rel err about 1e-6). `matmul.blk_k_mismatch` errors at trace time on both. The other 37 mutants fail in interpret mode and also fail
on hardware (37 of 37), so the hardware detection rate of the interpret-detectable mutants is unchanged; for `add.x_off_oob` the
interpret result is a wrong value and the hardware result is NaN.

## Tolerance fix (v3)
The oracle tolerance is `max|out - ref| / max|ref| < 2e-4`. On TPU the default matmul precision rounds f32 operands to bf16
(unit roundoff 2^-9 = 2e-3, 3.9e-3 worst case per operand) with f32 accumulation, so the unmodified base kernels `matmul.base`
(rel err 2.26e-3) and `attn.base` (2.96e-3) were marked FAIL natively. This is a property of the TPU, not a kernel bug.

What was changed in `pallas_tpu_suite.py`:
1. Dot-based families (`matmul`, `attn`) are also run natively with `precision=jax.lax.Precision.HIGHEST` in every dot, at the
   strict tolerance 2e-4. The native verdict is this run when it compiled and ran, else the default-precision run
   (`native_precision` field says which). Both runs are stored (`native_tpu_default`, `native_tpu_highest`).
2. A per-family tolerance for native default precision, `NATIVE_DEFAULT_TOL = {"attn": 5e-3}`: 1.7x the largest measured base error,
   1.3x the bf16 half-ulp, and 120x below the smallest attn mutant error on hardware (0.61).
3. The matmul family keeps the strict tolerance at default precision. Reason (checked on the v2 native data): any tolerance that
   accepts `matmul.base` (2.26e-3) also accepts `matmul.prec_bf16` (2.26e-3, explicit bf16 cast), because on TPU default precision
   the two compute the same thing. With the 5e-3 tolerance this is the only mutant of 45 whose verdict flips (FAIL to PASS); for attn
   no mutant flips. So a looser matmul tolerance would silently hide a detected mutant; precision HIGHEST is the way to
   separate base from `prec_bf16`.
4. `classify()` has a new class `runtime_halt_oob` ("core halted", "continuator halted"): in v2 `add.grid_big_oob` was labelled
   `vmem_oom` by the keyword matcher although the error is a runtime halt.

Offline validation (CPU interpret, jax 0.11.1): all 51 interpret verdicts unchanged; `precision=HIGHEST` accepted by interpret mode
(matmul.base 3.1e-7, attn.base 1.7e-7, matmul.prec_bf16 still 2.26e-3). A mock of the native stage driven by the stored v2 numbers
shows: HIGHEST works -> both base kernels PASS, no mutant changes; HIGHEST unavailable -> only `attn.base` flips (FAIL to PASS), no
mutant flips.

## Run v3 (this tolerance fix, native)
Status at the time of writing: the v3 kernel (version 3 of `gemma4-app-c-tpu-tpu`, TPU v5e-8) was pushed on 2026-10-17 and was still
QUEUED after more than 80 minutes, so no v3 numbers exist yet. Expected outcome, from the offline mock: both base kernels PASS,
`matmul.prec_bf16` still detected, the six hardware-only mutants unchanged. This section must be replaced by the measured
`tpu-out-v3/` numbers (`python summarize.py tpu-out-v3/results.json`) before any statement about v3 is made public.

## Limits
* One TPU generation (v5e) and one libtpu/jax pair (jax 0.11.0, libtpu 0.0.17 after refresh). Other generations (v4, v6e, v7) have
  different tiling, VMEM size and matmul behaviour.
* 51 variants over 6 small kernel families, fixed shapes, f32 only, one seed; no bf16/int8 tiling, megacore, `dimension_semantics`,
  multi-chip or multi-host effects.
* The mutants are synthetic, hand-written edits of one operator each; they are not bugs harvested from real code. The percentage of
  hardware-only mutants (13%) depends on how many tiling/VMEM/OOB variants were included and must not be read as a rate in the wild.
* Single oracle tolerance per mode; raw errors are stored so other thresholds can be applied post hoc.
* Two of the five ERR cases are runtime halts of the TPU core; they were run in isolated child processes, and a halt can in
  principle affect later variants of the same child (none was observed).
* The libtpu refresh from PyPI is part of the run; results are for the refreshed libtpu, not for the stock Kaggle image.
