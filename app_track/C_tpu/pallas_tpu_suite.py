"""Pallas interpret-vs-TPU mutation suite (self-contained).
Parent mode (default): orchestrates child processes (one TPU client per child), writes results.json incrementally.
Child mode: --child OUTJSONL id1 id2 ...
Env: SUITE_BUDGET_SEC (default 1500), SUITE_OUT (output dir).
"""
import os, sys, json, time, subprocess, glob, traceback

OUT = os.environ.get("SUITE_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(os.getcwd(), "out"))
os.makedirs(OUT, exist_ok=True)
TOL = 2e-4  # max|o-ref| / max|ref|  (strict tolerance: interpret mode, and native TPU at HIGHEST matmul precision)
# Native-TPU tolerance for dot-based families at the TPU DEFAULT matmul precision.
# Derivation: with default precision the MXU rounds f32 operands to bf16 (8-bit significand, unit roundoff 2^-9 = 1.95e-3,
# worst-case 2^-8 = 3.9e-3 per operand) and accumulates in f32. For random inputs the errors of the K=256 dot average out;
# measured max|o-ref|/max|ref| on v5e (jax 0.11.0, libtpu 0.0.17): matmul.base 2.26e-3, attn.base 2.96e-3 (strict 2e-4 -> false FAIL).
# 5e-3 = about 1.7x the largest measured base error and 1.3x the bf16 half-ulp 3.9e-3, and 120x below the smallest attn mutant
# error on hardware (0.61). Checked against the v2 native results: with 5e-3 NO attn mutant flips to PASS. For matmul it is NOT used:
# matmul.prec_bf16 (explicit bf16 cast) is numerically identical to the default-precision base on hardware (both 2.26e-3), so any
# tolerance that accepts matmul.base also accepts prec_bf16 (a detected mutant would become undetected). The matmul family therefore
# keeps the strict 2e-4 and its native verdict comes from the HIGHEST-precision run (see child()); if that run is unavailable the
# matmul.base native FAIL is a documented false FAIL and prec_bf16 is natively equivalent at default precision.
NATIVE_DEFAULT_TOL = {"attn": 5e-3}   # per-family tolerance for native TPU at DEFAULT matmul precision; missing family -> TOL
DOT_FAMILIES = ("matmul", "attn")
PREC = None  # set to jax.lax.Precision.HIGHEST by run_one(mode="native_highest"); read at trace time inside build()

# ----------------------------------------------------------------- variants
# (id, family, operator, knobs)
V = []
def add(fam, name, op, **k): V.append((f"{fam}.{name}", fam, op, k))
for f in ["add", "matmul", "softmax", "layernorm", "rowsum", "attn"]:
    add(f, "base", "BASE")
# add
add("add", "x_off", "IDX", x_off=1)
add("add", "out_swap", "IDX", out_swap=1)
add("add", "blk_half", "BLK", blk_half=1)
add("add", "grid_small", "GRID", grid_small=1)
add("add", "blk_lane100", "BLK-TILING", blk_lane100=1)
# matmul
add("matmul", "x_swap", "IDX", x_swap=1)
add("matmul", "y_swap", "IDX", y_swap=1)
add("matmul", "blk_k_mismatch", "BLK", blk_k_mismatch=1)
add("matmul", "no_zero", "INIT", no_zero=1)
add("matmul", "init_one", "INIT", init_one=1)
add("matmul", "k_grid_1", "GRID", k_grid_1=1)
add("matmul", "k_outer", "GRID-ORDER", k_outer=1)
add("matmul", "guard_axis1", "GUARD", guard_axis1=1)
add("matmul", "guard_always", "GUARD", guard_always=1)
add("matmul", "pid_axis0", "PID", pid_axis0=1)
add("matmul", "prec_bf16", "ARITH", prec_bf16=1)
# softmax
add("softmax", "idx_off", "IDX", idx_off=1)
add("softmax", "blk_4rows", "BLK-TILING", blk_4rows=1)
add("softmax", "axis0_max", "RED", axis0_max=1)
add("softmax", "sum_axis0", "RED", sum_axis0=1)
add("softmax", "no_max", "ARITH", no_max=1)
add("softmax", "no_norm", "ARITH", no_norm=1)
# layernorm
add("layernorm", "mean_axis0", "RED", mean_axis0=1)
add("layernorm", "var_nomean", "ARITH", var_nomean=1)
add("layernorm", "mul_sqrt", "ARITH", mul_sqrt=1)
add("layernorm", "no_eps", "ARITH", no_eps=1)
# rowsum (ragged)
add("rowsum", "no_mask", "MASK", no_mask=1)
add("rowsum", "mask_off", "MASK", mask_off=1)
add("rowsum", "no_init", "INIT", no_init=1)
add("rowsum", "guard_wrong", "GUARD", guard_wrong=1)
add("rowsum", "red_axis0", "RED", red_axis0=1)
add("rowsum", "grid_small", "GRID", grid_small=1)
add("rowsum", "pid_swap", "PID", pid_swap=1)
# attention
add("attn", "q_off", "IDX", q_off=1)
add("attn", "kmap_h0", "IDX", kmap_h0=1)
add("attn", "causal_none", "MASK", causal_none=1)
add("attn", "causal_lt", "MASK", causal_lt=1)
add("attn", "softmax_axis0", "RED", softmax_axis0=1)
add("attn", "mask_pid0", "PID", mask_pid0=1)
add("attn", "no_scale", "ARITH", no_scale=1)
add("attn", "v_is_k", "ARITH", v_is_k=1)
# late / runtime-risky (OOB indices, VMEM)
add("add", "x_off_oob", "IDX-OOB", x_off_oob=1)
add("add", "grid_big_oob", "GRID-OOB", grid_big_oob=1)
add("layernorm", "g_map_i_oob", "IDX-OOB", g_map_i=1)
add("add", "huge_block_vmem", "BLK-VMEM", huge=1)
VID = {v[0]: v for v in V}

# ----------------------------------------------------------------- families
def build(fam, k):
    """returns (kernel_call(jnp inputs, interpret)->array, numpy inputs, numpy ref)"""
    import numpy as np, jax, jax.numpy as jnp
    from jax.experimental import pallas as pl
    BS = pl.BlockSpec
    def R(seed, shape, scale=1.0):
        return (np.random.RandomState(seed).randn(*shape) * scale).astype(np.float32)
    f32 = jnp.float32

    if fam == "add":
        if k.get("huge"):
            shp = (4096, 2048); x, y = R(1, shp), R(2, shp)
            def call(a, b, interp):
                def kern(x_ref, y_ref, o_ref): o_ref[...] = x_ref[...] + y_ref[...]
                return pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct(shp, f32), grid=(1, 1),
                    in_specs=[BS(shp, lambda i, j: (0, 0))] * 2, out_specs=BS(shp, lambda i, j: (0, 0)), interpret=interp)(a, b)
            return call, (x, y), x + y
        shp = (256, 512); x, y = R(1, shp), R(2, shp)
        bs = (64, 128); grid = (4, 4)
        if k.get("blk_half"): bs = (32, 128)
        if k.get("grid_small"): grid = (3, 4)
        if k.get("grid_big_oob"): grid = (5, 4)
        if k.get("blk_lane100"): bs = (64, 100); grid = (4, 6)
        xm = lambda i, j: (i, j)
        if k.get("x_off"): xm = lambda i, j: (i, (j + 1) % 4)
        if k.get("x_off_oob"): xm = lambda i, j: (i, j + 1)
        om = lambda i, j: (j, i) if k.get("out_swap") else (i, j)
        def call(a, b, interp):
            def kern(x_ref, y_ref, o_ref): o_ref[...] = x_ref[...] + y_ref[...]
            return pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct(shp, f32), grid=grid,
                in_specs=[BS(bs, xm), BS(bs, lambda i, j: (i, j))], out_specs=BS(bs, om), interpret=interp)(a, b)
        return call, (x, y), x + y

    if fam == "matmul":
        M = N = K = 256; B = 128
        x, y = R(3, (M, K)), R(4, (K, N))
        ref = x.astype(np.float64) @ y.astype(np.float64)
        grid = (2, 2, 2)
        kpos = 2  # axis index of k in grid
        if k.get("k_grid_1"): grid = (2, 2, 1)
        xbs = (B, B)
        if k.get("blk_k_mismatch"): xbs = (B, B // 2)
        if k.get("k_outer"):
            kpos = 0
            xm = lambda kk, i, j: (i, kk); ym = lambda kk, i, j: (kk, j); om = lambda kk, i, j: (i, j)
        else:
            xm = lambda i, j, kk: (i, kk); ym = lambda i, j, kk: (kk, j); om = lambda i, j, kk: (i, j)
            if k.get("x_swap"): xm = lambda i, j, kk: (kk, i)
            if k.get("y_swap"): ym = lambda i, j, kk: (j, kk)
        def call(a, b, interp):
            def kern(x_ref, y_ref, o_ref):
                gaxis = kpos
                if k.get("guard_axis1"): gaxis = 1
                if k.get("pid_axis0"): gaxis = 0
                xx, yy = x_ref[...], y_ref[...]
                if k.get("prec_bf16"):
                    xx, yy = xx.astype(jnp.bfloat16), yy.astype(jnp.bfloat16)
                prod = jnp.dot(xx, yy, preferred_element_type=f32, precision=PREC)
                if k.get("guard_always"):
                    o_ref[...] = prod
                    return
                @pl.when(pl.program_id(gaxis) == 0)
                def _():
                    if k.get("no_zero"): pass
                    elif k.get("init_one"): o_ref[...] = jnp.ones_like(o_ref)
                    else: o_ref[...] = jnp.zeros_like(o_ref)
                o_ref[...] += prod
            return pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct((M, N), f32), grid=grid,
                in_specs=[BS(xbs, xm), BS((B, B), ym)], out_specs=BS((B, B), om), interpret=interp)(a, b)
        return call, (x, y), ref

    if fam == "softmax":
        R_, C = 64, 256
        x = R(5, (R_, C), 40.0)
        z = x.astype(np.float64); z = z - z.max(-1, keepdims=True); e = np.exp(z); ref = e / e.sum(-1, keepdims=True)
        rb = 4 if k.get("blk_4rows") else 8
        grid = (R_ // rb,)
        im = (lambda i: ((i + 1) % grid[0], 0)) if k.get("idx_off") else (lambda i: (i, 0))
        def call(a, interp):
            def kern(x_ref, o_ref):
                v = x_ref[...]
                if not k.get("no_max"):
                    v = v - v.max(axis=0 if k.get("axis0_max") else -1, keepdims=True)
                e = jnp.exp(v)
                if k.get("no_norm"): o_ref[...] = e
                else: o_ref[...] = e / e.sum(axis=0 if k.get("sum_axis0") else -1, keepdims=True)
            return pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct((R_, C), f32), grid=grid,
                in_specs=[BS((rb, C), im)], out_specs=BS((rb, C), lambda i: (i, 0)), interpret=interp)(a)
        return call, (x,), ref

    if fam == "layernorm":
        R_, C = 64, 256
        x = R(6, (R_, C)) * 2 + 3; g = R(7, (1, C)); b = R(8, (1, C))
        xd = x.astype(np.float64); m = xd.mean(-1, keepdims=True); v = ((xd - m) ** 2).mean(-1, keepdims=True)
        ref = (xd - m) / np.sqrt(v + 1e-5) * g + b
        gm = (lambda i: (i, 0)) if k.get("g_map_i") else (lambda i: (0, 0))
        def call(a, gg, bb, interp):
            def kern(x_ref, g_ref, b_ref, o_ref):
                v_ = x_ref[...]
                m_ = v_.mean(axis=0 if k.get("mean_axis0") else -1, keepdims=True)
                if k.get("var_nomean"): var = (v_ * v_).mean(-1, keepdims=True)
                else: var = ((v_ - m_) ** 2).mean(-1, keepdims=True)
                eps = 0.0 if k.get("no_eps") else 1e-5
                if k.get("mul_sqrt"): sc = jnp.sqrt(var + eps)
                else: sc = jax.lax.rsqrt(var + eps)
                o_ref[...] = (v_ - m_) * sc * g_ref[...] + b_ref[...]
            return pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct((R_, C), f32), grid=(R_ // 8,),
                in_specs=[BS((8, C), lambda i: (i, 0)), BS((1, C), gm), BS((1, C), lambda i: (0, 0))],
                out_specs=BS((8, C), lambda i: (i, 0)), interpret=interp)(a, gg, bb)
        return call, (x, g, b), ref

    if fam == "rowsum":
        R_, C, RB, CB = 256, 200, 64, 128
        x = R(9, (R_, C)) + 1.0
        ref = np.broadcast_to(x.astype(np.float64).sum(-1, keepdims=True), (R_, 128))
        grid = (R_ // RB, 1 if k.get("grid_small") else 2)
        def call(a, interp):
            def kern(x_ref, o_ref):
                i0 = pl.program_id(0)
                j = pl.program_id(0) if k.get("pid_swap") else pl.program_id(1)
                v = x_ref[...]
                if not k.get("no_mask"):
                    cols = j * CB + jax.lax.broadcasted_iota(jnp.int32, (RB, CB), 1)
                    lim = C + 1 if k.get("mask_off") else C
                    v = jnp.where(cols < lim, v, 0.0)
                gj = 1 if k.get("guard_wrong") else 0
                @pl.when(pl.program_id(1) == gj)
                def _():
                    if not k.get("no_init"): o_ref[...] = jnp.zeros_like(o_ref)
                if k.get("red_axis0"): s = jnp.sum(v, axis=0, keepdims=True)
                else: s = jnp.sum(v, axis=1, keepdims=True)
                o_ref[...] += s
            return pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct((R_, 128), f32), grid=grid,
                in_specs=[BS((RB, CB), lambda i, j: (i, j))], out_specs=BS((RB, 128), lambda i, j: (i, 0)),
                interpret=interp)(a)
        return call, (x,), ref

    if fam == "attn":
        H, S, D, BQ = 2, 128, 128, 64
        q, kx, vx = R(10, (H, S, D)), R(11, (H, S, D)), R(12, (H, S, D))
        sc = np.einsum("hqd,hkd->hqk", q.astype(np.float64), kx.astype(np.float64)) / np.sqrt(D)
        msk = np.tril(np.ones((S, S), bool)); sc = np.where(msk, sc, -1e30)
        sc = sc - sc.max(-1, keepdims=True); p = np.exp(sc); p /= p.sum(-1, keepdims=True)
        ref = np.einsum("hqk,hkd->hqd", p, vx.astype(np.float64))
        qm = (lambda h, qi: (h, (qi + 1) % 2, 0)) if k.get("q_off") else (lambda h, qi: (h, qi, 0))
        km = (lambda h, qi: (0, 0, 0)) if k.get("kmap_h0") else (lambda h, qi: (h, 0, 0))
        def call(a, b, c, interp):
            def kern(q_ref, k_ref, v_ref, o_ref):
                qv, kv = q_ref[0], k_ref[0]
                vv = kv if k.get("v_is_k") else v_ref[0]
                s = jax.lax.dot_general(qv, kv, (((1,), (1,)), ((), ())), preferred_element_type=f32, precision=PREC)
                if not k.get("no_scale"): s = s / (D ** 0.5)
                if not k.get("causal_none"):
                    qi = pl.program_id(0) if k.get("mask_pid0") else pl.program_id(1)
                    row = qi * BQ + jax.lax.broadcasted_iota(jnp.int32, (BQ, S), 0)
                    col = jax.lax.broadcasted_iota(jnp.int32, (BQ, S), 1)
                    ok = (col < row) if k.get("causal_lt") else (col <= row)
                    s = jnp.where(ok, s, -1e30)
                ax = 0 if k.get("softmax_axis0") else -1
                s = s - s.max(axis=ax, keepdims=True)
                e = jnp.exp(s); p_ = e / e.sum(axis=ax, keepdims=True)
                o_ref[0] = jnp.dot(p_, vv, preferred_element_type=f32, precision=PREC)
            return pl.pallas_call(kern, out_shape=jax.ShapeDtypeStruct((H, S, D), f32), grid=(H, S // BQ),
                in_specs=[BS((1, BQ, D), qm), BS((1, S, D), km), BS((1, S, D), lambda h, qi: (h, 0, 0))],
                out_specs=BS((1, BQ, D), lambda h, qi: (h, qi, 0)), interpret=interp)(a, b, c)
        return call, (q, kx, vx), ref
    raise KeyError(fam)

def classify(e):
    s = (type(e).__name__ + ": " + str(e)).lower()
    if "halted unexpectedly" in s or "core halt" in s or "continuator" in s: return "runtime_halt_oob"
    if "vmem" in s or "resource_exhausted" in s or "out of memory" in s or "scoped" in s: return "vmem_oom"
    if "divisible" in s or "tiling" in s or "block shape" in s or "block_shape" in s or "(8, 128)" in s or "8 and 128" in s: return "tiling_constraint"
    if "not implemented" in s or "unsupported" in s or "not supported" in s or "unimplemented" in s: return "unsupported_op"
    if "mosaic" in s or "lowering" in s or "tpu custom call" in s: return "mosaic_compile_other"
    if "shape" in s or "dimension" in s or "incompatible" in s or "broadcast" in s: return "shape_mismatch_trace"
    return "other:" + type(e).__name__

def run_one(vid, mode):
    """mode: interpret | native (TPU default matmul precision) | native_highest (jax.lax.Precision.HIGHEST in every dot)"""
    import numpy as np, jax, jax.numpy as jnp
    global PREC
    _, fam, op, k = VID[vid]
    r = dict(status="?", sec=None)
    t = time.time()
    PREC = jax.lax.Precision.HIGHEST if mode == "native_highest" else None
    tol = NATIVE_DEFAULT_TOL.get(fam, TOL) if mode == "native" else TOL
    try:
        call, inputs, ref = build(fam, k)
        if mode == "interpret":
            cpu = jax.devices("cpu")[0]
            with jax.default_device(cpu):
                arrs = [jax.device_put(a, cpu) for a in inputs]
                out = np.asarray(jax.block_until_ready(jax.jit(lambda *a: call(*a, True))(*arrs)))
        else:
            arrs = [jnp.asarray(a) for a in inputs]
            out = np.asarray(jax.block_until_ready(jax.jit(lambda *a: call(*a, False))(*arrs)))
        refm = float(np.abs(ref).max())
        diff = np.abs(out.astype(np.float64) - ref)
        nan = bool(np.isnan(out).any())
        mad = float(np.nanmax(diff)) if not nan or not np.isnan(diff).all() else float("nan")
        rel = (float("inf") if nan else mad / max(refm, 1e-12))
        r.update(status="ran", max_abs_err=None if mad != mad else mad, ref_max=refm, rel_err=None if rel == float("inf") else rel,
                 has_nan=nan, tol=tol, passed=bool((not nan) and rel < tol), out_shape=list(out.shape))
    except BaseException as e:
        r.update(status="error", err_class=classify(e), err=(type(e).__name__ + ": " + str(e))[:500].replace("\n", " | "))
    r["sec"] = round(time.time() - t, 2)
    return r

def child(outpath, ids):
    import jax
    plat = jax.default_backend()
    for vid in ids:
        with open(outpath, "a") as f: f.write(json.dumps(dict(id=vid, stage="start")) + "\n")
        rec = dict(id=vid, family=VID[vid][1], operator=VID[vid][2], backend=plat)
        rec["interpret_cpu"] = run_one(vid, "interpret")
        if plat == "tpu":
            nd = run_one(vid, "native")          # TPU default matmul precision (tol 5e-3 for attn, else strict)
            rec["native_tpu_default"] = nd
            if VID[vid][1] in DOT_FAMILIES:
                nh = run_one(vid, "native_highest")   # strict tol 2e-4
                rec["native_tpu_highest"] = nh
                # verdict rule: HIGHEST if it compiled and ran, else the default-precision run with the documented tolerance
                if nh.get("status") == "ran":
                    rec["native_tpu"], rec["native_precision"] = nh, "highest"
                else:
                    rec["native_tpu"], rec["native_precision"] = nd, "default"
            else:
                rec["native_tpu"], rec["native_precision"] = nd, "default"
        else:
            rec["native_tpu"] = dict(status="skipped")
        rec["stage"] = "done"
        with open(outpath, "a") as f: f.write(json.dumps(rec) + "\n")

def env_info():
    info = dict(python=sys.version.split()[0])
    try:
        import jax, jaxlib
        info.update(jax=jax.__version__, jaxlib=jaxlib.__version__, backend=jax.default_backend(),
                    devices=[str(d) for d in jax.devices()], device_kind=[d.device_kind for d in jax.devices()][:1])
    except Exception as e:
        info["jax_error"] = repr(e)
    try:
        import importlib.metadata as m
        for p in ["libtpu", "libtpu-nightly", "jax-tpu", "numpy", "ml_dtypes"]:
            try: info["pkg_" + p] = m.version(p)
            except Exception: pass
    except Exception: pass
    return info

def main():
    t0 = time.time(); budget = float(os.environ.get("SUITE_BUDGET_SEC", "1500"))
    jsonl = os.path.join(OUT, "variants.jsonl")
    if os.path.exists(jsonl): os.remove(jsonl)
    # environment probe in a subprocess (keeps parent TPU-free)
    probe = subprocess.run([sys.executable, __file__, "--env"], capture_output=True, text=True)
    print("ENV PROBE:", probe.stdout[-1500:], probe.stderr[-1500:], flush=True)
    try: env = json.loads(probe.stdout.strip().splitlines()[-1])
    except Exception: env = {"probe_failed": (probe.stdout + probe.stderr)[-1500:]}
    print("accel devices:", glob.glob("/dev/accel*"), glob.glob("/dev/vfio/*"), flush=True)
    if env.get("backend") != "tpu" and (glob.glob("/dev/accel*") or glob.glob("/dev/vfio/[0-9]*")) and os.environ.get("SUITE_TRY_PIP", "1") == "1":
        print("TPU hardware present but jax backend is", env.get("backend"), "-> pip install jax[tpu]", flush=True)
        p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "jax[tpu]", "-f",
                            "https://storage.googleapis.com/jax-releases/libtpu_releases.html"], capture_output=True, text=True)
        print("pip:", (p.stdout + p.stderr)[-1500:], flush=True)
        probe = subprocess.run([sys.executable, __file__, "--env"], capture_output=True, text=True)
        print("ENV PROBE2:", probe.stdout[-1500:], probe.stderr[-1500:], flush=True)
        try: env = json.loads(probe.stdout.strip().splitlines()[-1])
        except Exception: pass
    # --- libtpu refresh: the Kaggle TPU image ships a libtpu too old for jax's Pallas TPU ("requires a libtpu version
    # that's at most a month old"). Smoke-test one native kernel; on that error try newer libtpu wheels and point
    # TPU_LIBRARY_PATH at the new .so (children inherit os.environ). All steps are logged into env["libtpu_fix"].
    def _native_smoke():
        tmp = os.path.join(OUT, "smoke.jsonl")
        if os.path.exists(tmp): os.remove(tmp)
        try:
            subprocess.run([sys.executable, __file__, "--child", tmp, "add.base"], capture_output=True, text=True, timeout=300)
            for line in open(tmp):
                d = json.loads(line)
                if d.get("stage") == "done":
                    return (d.get("native_tpu") or {})
        except Exception as e:
            return {"status": "error", "err": "smoke failed: " + repr(e)[:300]}
        return {"status": "error", "err": "no smoke result"}
    def _libtpu_so():
        try:
            import importlib.util
            spec = importlib.util.find_spec("libtpu")
            so = os.path.join(os.path.dirname(spec.origin), "libtpu.so") if spec and spec.origin else None
            return so if so and os.path.exists(so) else None
        except Exception:
            return None
    fix_log = []
    if env.get("backend") == "tpu" and os.environ.get("SUITE_LIBTPU_FIX", "1") == "1":
        sm = _native_smoke(); fix_log.append(dict(step="smoke0", **{k: sm.get(k) for k in ("status", "passed", "err")}))
        print("NATIVE SMOKE 0:", sm, flush=True)
        if sm.get("status") == "error" and "libtpu" in str(sm.get("err", "")):
            for pkgs in (["libtpu"], ["libtpu-nightly", "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"]):
                p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + pkgs, capture_output=True, text=True, timeout=900)
                so = _libtpu_so()
                if so: os.environ["TPU_LIBRARY_PATH"] = so
                sm = _native_smoke()
                fix_log.append(dict(step="pip " + " ".join(pkgs), rc=p.returncode, so=so, tail=(p.stdout + p.stderr)[-400:],
                                    **{k: sm.get(k) for k in ("status", "passed", "err")}))
                print("LIBTPU FIX", pkgs, "so=", so, "smoke=", sm, flush=True)
                if sm.get("status") == "ran":
                    break
    env["libtpu_fix"] = fix_log
    results = {}; crashes = []
    remaining = [v[0] for v in V]
    while remaining and time.time() - t0 < budget:
        left = budget - (time.time() - t0)
        p = subprocess.Popen([sys.executable, __file__, "--child", jsonl] + remaining, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            outtxt, _ = p.communicate(timeout=max(30, min(left, 900)))
            rc = p.returncode
        except subprocess.TimeoutExpired:
            p.kill(); outtxt, _ = p.communicate(); rc = "timeout"
        print("child rc", rc, "tail:", (outtxt or "")[-600:], flush=True)
        started = None
        if os.path.exists(jsonl):
            for line in open(jsonl):
                try: d = json.loads(line)
                except Exception: continue
                if d.get("stage") == "done": results[d["id"]] = d; started = None
                elif d.get("stage") == "start": started = d["id"]
        remaining = [r for r in remaining if r not in results]
        if remaining and rc != 0:
            bad = started if started in remaining else remaining[0]
            results[bad] = dict(id=bad, family=VID[bad][1], operator=VID[bad][2], backend=env.get("backend"),
                                interpret_cpu=dict(status="unknown_after_crash"),
                                native_tpu=dict(status="process_crash", err_class="process_crash_or_hang", err=str(rc) + " " + (outtxt or "")[-300:].replace("\n", " | ")))
            crashes.append(bad); remaining.remove(bad)
        elif remaining and rc == 0:
            break
        with open(os.path.join(OUT, "results.json"), "w") as f:
            json.dump(dict(env=env, tol=TOL, native_default_tol=NATIVE_DEFAULT_TOL, dot_families=list(DOT_FAMILIES), elapsed=round(time.time() - t0), crashes=crashes, results=list(results.values()), unrun=remaining), f, indent=1)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(dict(env=env, tol=TOL, native_default_tol=NATIVE_DEFAULT_TOL, dot_families=list(DOT_FAMILIES), elapsed=round(time.time() - t0), crashes=crashes, results=list(results.values()), unrun=remaining), f, indent=1)
    # summary
    def cls(r):
        if r.get("status") == "ran": return "PASS" if r["passed"] else "FAIL"
        return "ERR"
    print("\nid | interp | tpu")
    for r in results.values():
        print(r["id"], cls(r["interpret_cpu"]) if r["interpret_cpu"].get("status") != "unknown_after_crash" else "?", "|",
              cls(r["native_tpu"]) if r["native_tpu"].get("status") in ("ran", "error") else r["native_tpu"].get("status"),
              r["native_tpu"].get("err_class", ""))
    print("DONE elapsed", round(time.time() - t0), "unrun", remaining, flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--env":
        print(json.dumps(env_info()))
    elif len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3:])
    else:
        main()
