#!/usr/bin/env python3
"""Reproducibility smoke test for the release package (CPU only, no network, no competition data).

Run from the package root:   python smoke_test.py [--quick] [--json out.json]

Checks
  1. A/B analysis kit end to end on its synthetic generator (analyze_ab.py CLI, tar + dir layouts).
  2. The 36 kit unit tests (skipped with --quick; ~2.5 min; the 1 real-fixture test is skipped by design).
  3. Re-derivation of the session-1 numbers (B11, B13) from the scrubbed checkpoint rows.
  4. Synthetic task factory unit tests (synth/tests).
  5. Pallas mutation factory: mutation operators on the 19 original seeds (pure ast, no jax). If jax is installed
     and the platform allows (needs the POSIX `resource` module), also verify mutants of 2 seeds by execution.
  6. Pre-registered study A: the analysis on the shipped numbers-only features reproduces AUC gain +0.002.
  7. Strategy simulations s1/s2 re-run (inputs inline).
  8. Leak scan (only if a blocklist is supplied through $LEAK_BLOCKLIST).
Exit status 0 iff every executed check passed.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RESULTS = []


def rec(name, ok, detail="", t0=None):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:400], "sec": round(time.time() - t0, 1) if t0 else None})
    print(("PASS " if ok else "FAIL ") + name + ("  " + str(detail)[:200] if detail else ""), flush=True)


def sh(args, cwd=ROOT, timeout=900, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=e)


def check_ab_synthetic():
    t0 = time.time()
    sys.path.insert(0, os.path.join(ROOT, "ab_kit"))
    import synth
    with tempfile.TemporaryDirectory() as td:
        rows, tr = synth.simulate(30, seed=7, effects={"lean": dict(tok_factor=0.6, edit_delta=-3)})
        synth.write_dataset(os.path.join(td, "tar"), rows, tr, tar=True)
        r = sh([PY, "ab_kit/analyze_ab.py", "--results", os.path.join(td, "tar"), "--traces",
                os.path.join(td, "tar", "traces_session1.tar.gz"), "--n-boot", "300", "--preflight", "pass",
                "--out-json", os.path.join(td, "o.json"), "--out-md", os.path.join(td, "o.md")])
        ok = r.returncode == 0 and os.path.exists(os.path.join(td, "o.json"))
        det = ""
        if ok:
            o = json.load(open(os.path.join(td, "o.json")))
            det = "n_complete=%s decision=%s" % (o.get("accounting", {}).get("n_complete"), str(o.get("decision"))[:60])
            ok = ok and o.get("accounting", {}).get("n_complete") == 30
        else:
            det = r.stderr[-300:]
    rec("ab_kit: synthetic world -> analyze_ab.py CLI", ok, det, t0)


def check_ab_tests():
    t0 = time.time()
    r = sh([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "ab_kit/test_analyze_ab.py"], timeout=1800)
    m = re.search(r"(\d+) passed", r.stdout)
    sk = re.search(r"(\d+) skipped", r.stdout)
    fl = re.search(r"(\d+) failed", r.stdout)
    rec("ab_kit: pytest (36 tests)", r.returncode == 0 and not fl, f"passed={m.group(1) if m else '?'} skipped={sk.group(1) if sk else 0}", t0)


def check_calc():
    t0 = time.time()
    r = sh([PY, "calc/ab_session1.py"])
    out = r.stdout
    ok = (r.returncode == 0 and "complete triples 13" in out and "control(v6) {'n': 13, 'resolved': 6, 'timeouts': 8" in out
          and "decide {'n': 13, 'resolved': 7, 'timeouts': 7" in out and "lean {'n': 13, 'resolved': 6, 'timeouts': 7" in out
          and "ratio-1 -0.177 CI -0.315 -0.058" in out and "ratio-1 -0.049 CI -0.278 0.171" in out)
    rec("paper numbers B11/B13 from scrubbed checkpoint rows", ok, out.strip().splitlines()[0] if out else r.stderr[-300:], t0)


def check_analyze_on_real_rows():
    t0 = time.time()
    r = sh([PY, "ab_kit/analyze_ab.py", "--results", "data/ckpt_scrubbed", "--launch-ids", "lbeval-20261007-043225",
            "--n-boot", "2000", "--preflight", "unknown"])
    ok = r.returncode == 0 and "CONTINUE" in r.stdout and "13/43" in r.stdout
    rec("B4: analyze_ab.py on scrubbed rows -> CONTINUE (13/43 complete pairs)", ok, "" if ok else (r.stdout[-200:] + r.stderr[-200:]), t0)


def check_synth_tests():
    t0 = time.time()
    r = sh([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "synth/tests"], timeout=900)
    m = re.search(r"(\d+) passed", r.stdout)
    rec("synth factory: unit tests", r.returncode == 0, f"passed={m.group(1) if m else '?'} " + ("" if r.returncode == 0 else r.stdout[-300:]), t0)


def check_pallas():
    t0 = time.time()
    d = os.path.join(ROOT, "app_track", "C_pallas")
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("pallas_mutate", os.path.join(d, "mutate.py"))
    mut = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mut)
    import ast
    seeds = sorted(f for f in os.listdir(os.path.join(d, "seeds")) if f.endswith(".py"))
    tot, bad, per_op = 0, 0, {}
    for f in seeds:
        src = open(os.path.join(d, "seeds", f), encoding="utf8").read()
        sites = mut.find_sites(src)
        for s in sites:
            try:
                ast.parse(mut.apply_site(src, s))
            except SyntaxError:
                bad += 1
            tot += 1
            per_op[s.get("op", "?") if isinstance(s, dict) else "?"] = per_op.get(s.get("op", "?") if isinstance(s, dict) else "?", 0) + 1
    rec("pallas factory: mutation operators on %d seeds" % len(seeds), len(seeds) >= 19 and tot > 100 and bad == 0,
        f"mutation sites={tot} unparsable={bad}", t0)
    try:
        import jax  # noqa: F401
        import resource  # noqa: F401
        have = True
    except Exception as e:
        have = False
        rec("pallas factory: execution-verified mutants (2 seeds)", True, "SKIPPED (jax and POSIX resource module required): %s" % str(e)[:80])
    if have:
        t1 = time.time()
        with tempfile.TemporaryDirectory() as td:
            sd = os.path.join(td, "seeds")
            os.makedirs(sd)
            for f in ("k01_vadd.py", "k03_softmax.py"):
                open(os.path.join(sd, f), "w").write(open(os.path.join(d, "seeds", f)).read())
            r = sh([PY, os.path.join(d, "run_factory.py"), sd, os.path.join(td, "out"), "3", "0"], timeout=1200)
            rows = os.path.join(td, "out", "rows.jsonl")
            n = sum(1 for _ in open(rows)) if os.path.exists(rows) else 0
            rec("pallas factory: execution-verified mutants (2 seeds)", r.returncode == 0 and n > 0, f"rows={n}", t1)


def check_study_a():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        r = sh([PY, "app_track/A_difficulty/analyze.py", "app_track/A_difficulty/results/features_tasks.jsonl", td], timeout=1800)
        res = os.path.join(td, "results.json")
        ok = os.path.exists(res)
        det = ""
        if ok:
            p = json.load(open(res))["primary"]
            det = "gain=%.3f ci=[%.3f, %.3f] %s" % (p["gain"], p["ci"][0], p["ci"][1], p.get("verdict"))
            ok = abs(p["gain"] - 0.0020) < 0.0015 and p["ci"][0] < 0 < p["ci"][1]
        else:
            det = "rc=%s " % r.returncode + (r.stdout[-200:] + r.stderr[-300:])
    rec("study A (pre-registered): gain +0.002 AUC, CI spans 0", ok, det, t0)


def check_strategy():
    t0 = time.time()
    ok, det = True, []
    with tempfile.TemporaryDirectory() as td:
        for s in ("s1_noise.py", "s2_posterior.py"):
            open(os.path.join(td, s), "w").write(open(os.path.join(ROOT, "strategy", s)).read())
            r = sh([PY, s], cwd=td, timeout=600)
            ok = ok and r.returncode == 0
            det.append(s + (":ok" if r.returncode == 0 else ":" + r.stderr[-120:]))
        if ok:
            a = json.load(open(os.path.join(td, "s1_noise.json")))
            b = json.load(open(os.path.join(ROOT, "strategy", "s1_noise.json")))
            ok = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
            det.append("s1 identical to shipped output" if ok else "s1 differs from shipped output")
    rec("strategy simulations s1,s2 re-run", ok, "; ".join(det), t0)


def check_scan():
    bl = os.environ.get("LEAK_BLOCKLIST")
    if not bl or not os.path.exists(bl):
        rec("leak scan", True, "SKIPPED (no $LEAK_BLOCKLIST; blocklist is private to the author)")
        return
    t0 = time.time()
    r = sh([PY, "tools/scan_competition_leaks.py", "scan", ".", "--blocklist", bl, "--quiet"])
    rec("leak scan of the package", r.returncode == 0, r.stdout[-200:], t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 36 kit unit tests and study A (about 1 minute)")
    ap.add_argument("--json")
    a = ap.parse_args()
    check_calc()
    check_analyze_on_real_rows()
    check_ab_synthetic()
    if not a.quick:
        check_ab_tests()
    check_synth_tests()
    check_pallas()
    if not a.quick:
        check_study_a()
    check_strategy()
    check_scan()
    ok = all(r["ok"] for r in RESULTS)
    print("\nSUMMARY: %d/%d checks passed" % (sum(r["ok"] for r in RESULTS), len(RESULTS)))
    if a.json:
        json.dump(RESULTS, open(a.json, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
