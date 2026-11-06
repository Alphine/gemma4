"""Mutation factory runner (needs jax). Verifies every mutant by execution with pallas_call(interpret=True) on CPU.

usage: python run_factory.py <seeds_dir> <out_dir> [max_per_op_seed] [lib_cap]
"""
import zlib
import sys, os, json, time, signal, random, traceback, resource, sys, collections, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mutate

ATOL, RTOL = 1e-4, 1e-4
TIMEOUT = 120

LIB = {
    "L01_tpu_matmul": ("ops/tpu/matmul.py", '''

def make_cases():
    k = jax.random.key
    return [(jax.random.normal(k(0), (256, 256)), jax.random.normal(k(1), (256, 256))),
            (jax.random.normal(k(2), (128, 384)), jax.random.normal(k(3), (384, 256)))]


def op(x, y):
    return matmul(x, y, block_shape=(128, 128), block_k=128)


def reference(x, y):
    return x @ y
'''),
    "L02_tpu_flash_attention": ("ops/tpu/flash_attention.py", '''

def make_cases():
    k = jax.random.key
    out = []
    for causal in (False, True):
        out.append(tuple(jax.random.normal(k(i), (1, 2, 256, 64)) for i in range(3)) + (causal,))
    return out


def op(q, k, v, causal):
    return flash_attention(q, k, v, causal=causal, sm_scale=0.125)


def reference(q, k, v, causal):
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k) * 0.125
    if causal:
        n = q.shape[2]
        s = jnp.where(jnp.arange(n)[:, None] >= jnp.arange(n)[None, :], s, -1e30)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, -1), v)
'''),
    "L03_gpu_mha": ("ops/gpu/attention.py", '''

def make_cases():
    k = jax.random.key
    return [tuple(jax.random.normal(k(i), (1, 64, 2, 32)) for i in range(3)) + (c,) for c in (False, True)]


def op(q, k, v, causal):
    return mha(q, k, v, None, sm_scale=0.25, causal=causal, interpret=True)


def reference(q, k, v, causal):
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k) * 0.25
    if causal:
        n = q.shape[1]
        s = jnp.where(jnp.arange(n)[:, None] >= jnp.arange(n)[None, :], s, -1e30)
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(s, -1), v)
'''),
    "L04_gpu_softmax": ("ops/gpu/softmax.py", '''

def make_cases():
    k = jax.random.key
    return [(3.0 * jax.random.normal(k(0), (8, 128)),), (3.0 * jax.random.normal(k(1), (4, 100)),)]


def op(x):
    return softmax(x, axis=-1, interpret=True)


def reference(x):
    return jax.nn.softmax(x, axis=-1)
'''),
}


class TO(Exception):
    pass


def _alarm(*a):
    raise TO()


def load(src, name):
    import types
    m = types.ModuleType(name)
    m.__file__ = "<%s>" % name
    sys.modules[name] = m
    exec(compile(src, "<%s>" % name, "exec"), m.__dict__)
    return m.__dict__


def executed_lines(src, name):
    """Lines of the seed source executed while running its cases (coverage filter for mutation sites)."""
    lines = set()
    fn = "<%s>" % name

    def tr(frame, ev, arg):
        if frame.f_code.co_filename != fn:
            return None
        def lt(fr, e, a):
            if e == "line":
                lines.add(fr.f_lineno)
            return lt
        lines.add(frame.f_lineno)
        return lt
    ns = load(src, name)
    sys.settrace(tr)
    try:
        for args in ns["make_cases"]():
            ns["op"](*args)
    finally:
        sys.settrace(None)
    return lines


def run_cases(ns, cases, refs):
    """returns (status, diag). status in ok|numerics_fail|crash|shape|timeout"""
    worst = None
    for ci, (args, ref) in enumerate(zip(cases, refs)):
        out = np.asarray(ns["op"](*args))
        if out.shape != ref.shape:
            return "shape", {"case": ci, "out_shape": list(out.shape), "ref_shape": list(ref.shape)}
        bad = ~np.isclose(out, ref, atol=ATOL, rtol=RTOL)
        if bad.any():
            d = np.abs(out.astype(np.float64) - ref.astype(np.float64))
            fin = np.isfinite(out).all()
            b2 = bad.reshape(bad.shape[0], -1) if bad.ndim > 1 else bad.reshape(-1, 1)
            diag = dict(case=ci, frac_bad=float(bad.mean()), max_err=float(np.nanmax(d)) if np.isfinite(d).any() else None,
                        finite=bool(fin), rows_bad=float(b2.any(1).mean()), cols_bad=float(b2.any(0).mean()),
                        first_half_bad=float(b2[: max(1, b2.shape[0] // 2)].mean()),
                        last_block_only=bool(b2.any(1).nonzero()[0].min() >= int(0.75 * b2.shape[0])) if b2.any() else False,
                        out_zero_frac=float((out == 0).mean()), ndim=int(ref.ndim))
            if worst is None:
                worst = diag
    if worst is not None:
        return "numerics_fail", worst
    return "ok", {}


def eval_mutant(src, name, cases, refs):
    t = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(TIMEOUT)
    try:
        ns = load(src, name)
        st, diag = run_cases(ns, cases, refs)
    except TO:
        st, diag = "timeout", {}
    except Exception as e:
        st, diag = "crash", {"exc": type(e).__name__, "msg": str(e)[:160]}
    finally:
        signal.alarm(0)
    diag["sec"] = round(time.time() - t, 2)
    return st, diag


def seed_source(name, path_or_lib):
    if name in LIB:
        import jax.experimental.pallas as pl
        rel, adapter = LIB[name]
        base = open(os.path.join(os.path.dirname(pl.__file__), rel)).read()
        header = ""
        # modules with `from __future__`/relative imports are not used here
        return base, adapter
    return open(path_or_lib).read(), ""


def process_seed(args):
    name, path, outdir, max_per_op, lib_cap, rng_seed = args
    import jax
    import jax.numpy as jnp
    import jax.experimental.pallas as pl
    orig = pl.pallas_call

    def forced(*a, **k):
        k["interpret"] = True
        return orig(*a, **k)
    pl.pallas_call = forced
    import warnings
    warnings.filterwarnings("ignore")
    rows = []
    t0 = time.time()
    base, adapter = seed_source(name, path)
    full = base + adapter
    try:
        ns = load(full, name)
        cases = ns["make_cases"]()
        refs = []
        for a in cases:
            refs.append(np.asarray(ns["reference"](*a)))
        st, diag = run_cases(ns, cases, refs)
        if st != "ok":
            return [dict(seed=name, kind="seed_fail", status=st, diag=diag)], {"seed": name, "ok": False}
        cov = executed_lines(full, name)
    except Exception as e:
        return [dict(seed=name, kind="seed_error", error=repr(e)[:300], tb=traceback.format_exc()[-800:])], {"seed": name, "ok": False}
    nbase = base.count("\n") + 1
    sites = mutate.find_sites(base)
    # keep only sites whose statement ran
    live = [s for s in sites if any(l in cov for l in range(s["stmt_span"][0], s["stmt_span"][1] + 1))]
    rng = random.Random(rng_seed + zlib.crc32(name.encode()) % 1000)
    byop = collections.defaultdict(list)
    for s in live:
        byop[s["op"]].append(s)
    chosen = []
    for op, ss in byop.items():
        rng.shuffle(ss)
        chosen += ss[:max_per_op]
    if name in LIB and len(chosen) > lib_cap:
        rng.shuffle(chosen)
        chosen = chosen[:lib_cap]
    meta = dict(seed=name, ok=True, n_sites=len(sites), n_live=len(live), n_chosen=len(chosen),
                seed_sec=round(time.time() - t0, 1), lines=nbase)
    for i, s in enumerate(chosen):
        msrc = mutate.apply_site(base, s) + adapter
        st1, d1 = eval_mutant(msrc, name + "_m%d" % i, cases, refs)
        st2, d2 = eval_mutant(msrc, name + "_m%d" % i, cases, refs)
        rows.append(dict(seed=name, mid="%s#%d" % (name, i), kind="mutant", site=s, status=st1, status2=st2,
                         flaky=(st1 != st2), diag=d1, sec=d1.get("sec", 0) + d2.get("sec", 0),
                         rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)))
    meta["sec"] = round(time.time() - t0, 1)
    return rows, meta


def main():
    seeds_dir, outdir = sys.argv[1], sys.argv[2]
    max_per_op = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    lib_cap = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    os.makedirs(outdir, exist_ok=True)
    jobs = []
    for fn in sorted(os.listdir(seeds_dir)):
        if fn.endswith(".py"):
            jobs.append((fn[:-3], os.path.join(seeds_dir, fn), outdir, max_per_op, lib_cap, 17))
    for n in LIB:
        jobs.append((n, None, outdir, max_per_op, lib_cap, 17))
    t0 = time.time()
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", "4")), maxtasksperchild=1) as pool, \
            open(os.path.join(outdir, "rows.jsonl"), "w") as f:
        metas = []
        for rows, meta in pool.imap_unordered(process_seed, jobs):
            metas.append(meta)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print("seed done", meta, round(time.time() - t0), flush=True)
    json.dump(metas, open(os.path.join(outdir, "seed_meta.json"), "w"), indent=1)
    # library original sources (tiny, needed for offline analysis; Apache-2.0, jax version recorded)
    import jax, jax.experimental.pallas as pl
    ld = os.path.join(outdir, "lib_src")
    os.makedirs(ld, exist_ok=True)
    for n, (rel, ad) in LIB.items():
        open(os.path.join(ld, n + ".py"), "w").write(open(os.path.join(os.path.dirname(pl.__file__), rel)).read())
        open(os.path.join(ld, n + ".adapter.py"), "w").write(ad)
    json.dump({"jax": jax.__version__}, open(os.path.join(outdir, "versions.json"), "w"))
    print("TOTAL SEC", round(time.time() - t0))


if __name__ == "__main__":
    main()
