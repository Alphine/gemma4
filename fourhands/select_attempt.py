"""select_attempt.py - pick ONE attempt's diff and leave exactly that diff in /workspace (selector step of fourhands).

Usage (run_skill_script): args with the single key root = /workspace (optional; default autodetect).
Candidates: hand A = the working tree of /workspace itself; hands B.. = private copies /tmp/hand_b, /tmp/hand_c, ...
Per candidate (all read-only until the final apply): diff vs the harness baseline commit with protected files stripped,
compile of the changed .py files, import of the changed modules from the candidate's own tree, the ONE ranked test file
of the first changed source file run on the candidate AND on a pristine export of the baseline (new failures = regression),
and agreement = number of other candidates editing the same file within 6 lines of one of its hunks.
Score (max): nonempty, compiles, imports, no regression, tests green, agreement, smaller diff, hand A on a tie.
Applying a non-A winner: A's diff is saved, /workspace is reset to the baseline and the winner's diff applied; the result
is re-diffed and compared with the winner's diff (VERIFIED), otherwise A is restored.
Prints a compact table and one final line starting with SELECTED.  Never raises; always exits 0.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locate import find_root, list_py, module_of, parse_args, rank_tests  # noqa: E402

PROTECTED = re.compile(
    r"(^|/)(conftest\.py|pytest\.ini|pyproject\.toml|tox\.ini|setup\.cfg|sitecustomize\.py|usercustomize\.py)$"
    r"|\.pth$|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)(tests?|testing)/.*\.py$")
TEST_TIMEOUT = 40
T0 = time.time()


def git(path, *args, timeout=30, inp=None):
    try:
        r = subprocess.run(["git", "-C", path] + list(args), capture_output=True, timeout=timeout, input=inp)
        return r.returncode, r.stdout, r.stderr.decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return 99, b"", "%s: %s" % (type(e).__name__, e)


def base_ref(path):
    rc, _, _ = git(path, "rev-parse", "--verify", "-q", "_swegemma_baseline")
    return "_swegemma_baseline" if rc == 0 else "HEAD"


def raw_diff(path):
    git(path, "add", "-N", ".")
    rc, out, err = git(path, "diff", "--binary", base_ref(path), timeout=60)
    return out if rc == 0 else b""


def split_blocks(diff_bytes):
    txt = diff_bytes.decode("utf-8", "surrogateescape")
    parts = re.split(r"(?m)^(?=diff --git )", txt)
    return [p for p in parts if p.startswith("diff --git ")]


def block_path(block):
    m = re.match(r"diff --git a/(.*?) b/(.*)\n", block)
    return m.group(2) if m else ""


def filtered(diff_bytes):
    """-> (text, [paths], n_changed_lines, {path: [(start, end)]})"""
    keep, paths, nlines, ranges = [], [], 0, {}
    for b in split_blocks(diff_bytes):
        p = block_path(b)
        if not p or PROTECTED.search(p) or p.endswith((".pyc", ".pyo")) or "__pycache__/" in p:
            continue
        keep.append(b)
        paths.append(p)
        for l in b.split("\n"):
            if (l.startswith("+") and not l.startswith("+++")) or (l.startswith("-") and not l.startswith("---")):
                nlines += 1
        for m in re.finditer(r"(?m)^@@ -(\d+)(?:,(\d+))? ", b):
            s = int(m.group(1))
            n = int(m.group(2)) if m.group(2) is not None else 1
            ranges.setdefault(p, []).append((s, s + max(n, 1) - 1))
    return "".join(keep), paths, nlines, ranges


def run(cmd, cwd, timeout, env_extra=None):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(env_extra or {})
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout or b"").decode("utf-8", "replace") + (r.stderr or b"").decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        return 98, "%s: %s" % (type(e).__name__, e)


def pytest_counts(out):
    p = f = 0
    m = re.search(r"(\d+) passed", out)
    if m:
        p = int(m.group(1))
    m = re.search(r"(\d+) failed", out)
    if m:
        f = int(m.group(1))
    m = re.search(r"(\d+) errors?", out)
    if m:
        f += int(m.group(1))
    return p, f


def run_test(tree, test):
    rc, out = run([sys.executable, "-m", "pytest", test, "-x", "-q", "-p", "no:cacheprovider", "--color=no"], tree, TEST_TIMEOUT,
                  {"PYTHONPATH": tree})
    if rc is None:
        return {"timeout": True, "passed": 0, "failed": 0}
    p, f = pytest_counts(out)
    if rc not in (0, 1, 5) and p == 0 and f == 0:
        f = 1   # collection/usage crash
    return {"timeout": False, "passed": p, "failed": f}


def map_tmp(p):
    """Subprocess sandbox: /tmp is <sandbox>/tmp next to <sandbox>/workspace (containers: identity)."""
    return p


def find_copies(ws):
    cands = []
    for pat in ("/tmp/hand_*",):
        cands += glob.glob(pat)
    sib = os.path.join(os.path.dirname(ws.rstrip("/")), "tmp")
    cands += glob.glob(os.path.join(sib, "hand_*"))
    seen, out = set(), []
    for c in sorted(cands):
        rp = os.path.realpath(c)
        if rp in seen or not os.path.isdir(os.path.join(c, ".git")) and not os.path.isfile(os.path.join(c, ".git")):
            continue
        seen.add(rp)
        out.append(c)
    return out


def main(argv):
    opts, _ = parse_args(argv)
    ws = find_root(opts)
    r = opts.get("root")
    if r and os.path.isdir(os.path.join(r, ".git")):
        ws = os.path.abspath(r)
    names = [("A", ws)]
    for c in find_copies(ws):
        names.append((os.path.basename(c).replace("hand_", "").upper() or "X", c))
    cands = []
    for nm, path in names:
        rd = raw_diff(path)
        text, paths, nlines, ranges = filtered(rd)
        cands.append({"name": nm, "path": path, "raw": rd, "text": text, "paths": paths, "nlines": nlines,
                      "ranges": ranges, "nonempty": bool(paths)})
    # pristine export of the baseline for regression checks
    pristine = tempfile.mkdtemp(prefix="sel_pristine_")
    try:
        rc, tarball, _ = git(ws, "archive", base_ref(ws), timeout=60)
        if rc == 0 and tarball:
            subprocess.run(["tar", "-x", "-C", pristine], input=tarball, timeout=60)
    except Exception:  # noqa: BLE001
        pass
    files = list_py(ws)
    jobs = {}
    for c in cands:
        srcs = [p for p in c["paths"] if p.endswith(".py")]
        c["srcs"] = srcs
        c["compile"] = True
        for p in srcs:
            full = os.path.join(c["path"], p)
            try:
                compile(open(full, "rb").read(), full, "exec")
            except SyntaxError:
                c["compile"] = False
            except OSError:
                pass   # deleted file
        c["import"] = True
        c["import_note"] = ""
        c["test"] = None
        if srcs and c["compile"]:
            src = srcs[0]
            mod = module_of(src)
            if mod and not src.split("/")[0] in ("scripts", "script", "tools", "bin", "docs", "docs_src", "examples") and src not in ("setup.py", "noxfile.py"):
                code = ("import importlib, sys; m = importlib.import_module(%r); "
                        "print('FILE', getattr(m, '__file__', ''))" % mod)
                rc, out = run([sys.executable, "-c", code], c["path"], 20, {"PYTHONPATH": c["path"] + os.pathsep + os.path.join(c["path"], "src")})
                if rc != 0:
                    c["import"] = False
                    c["import_note"] = out.strip().splitlines()[-1][:120] if out.strip() else "import failed"
                else:
                    mm = re.search(r"FILE (.*)", out)
                    if mm and mm.group(1) and os.path.realpath(c["path"]) not in os.path.realpath(mm.group(1)):
                        c["import_note"] = "imported from another tree: " + mm.group(1)[:80]
            try:
                rk = rank_tests(c["path"], files, src, limit=1)
                c["test"] = rk[0][1] if rk else None
            except Exception:  # noqa: BLE001
                c["test"] = None
    tests = sorted({c["test"] for c in cands if c["test"] and c["nonempty"] and c["compile"] and c["import"]})
    results = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {}
        for t in tests:
            futs[("base", t)] = ex.submit(run_test, pristine, t)
        for c in cands:
            if c["test"] and c["nonempty"] and c["compile"] and c["import"]:
                futs[(c["name"], c["test"])] = ex.submit(run_test, c["path"], c["test"])
        for k, f in futs.items():
            try:
                results[k] = f.result(timeout=TEST_TIMEOUT + 15)
            except Exception:  # noqa: BLE001
                results[k] = {"timeout": True, "passed": 0, "failed": 0}
    for c in cands:
        base = results.get(("base", c["test"])) if c["test"] else None
        res = results.get((c["name"], c["test"])) if c["test"] else None
        c["res"], c["base"] = res, base
        c["regress"] = bool(res and base and not res["timeout"] and res["failed"] > base["failed"])
        c["green"] = bool(res and not res["timeout"] and res["failed"] == 0 and res["passed"] > 0)
        agree = 0
        for o in cands:
            if o is c or not o["nonempty"]:
                continue
            hit = False
            for p, rs in c["ranges"].items():
                for (s, e) in rs:
                    for (s2, e2) in o["ranges"].get(p, []):
                        if s - 6 <= e2 and s2 <= e + 6:
                            hit = True
            agree += hit
        c["agree"] = agree
    for i, c in enumerate(cands):
        c["key"] = (c["nonempty"], c["compile"], c["import"], not c["regress"], c["green"], c["agree"], -c["nlines"], -i)
    print("candidate table (A=/workspace in place):")
    for c in cands:
        t = c["res"]
        print("  %s: files=%s lines=%d compile=%s import=%s%s test=%s %s regress=%s agree=%d" % (
            c["name"], ",".join(c["paths"]) or "-", c["nlines"], c["compile"], c["import"],
            (" [%s]" % c["import_note"]) if c["import_note"] else "", c["test"] or "-",
            ("p%d/f%d%s" % (t["passed"], t["failed"], "/TIMEOUT" if t["timeout"] else "")) if t else "-",
            c["regress"], c["agree"]))
    best = max(cands, key=lambda c: c["key"])
    shutil.rmtree(pristine, ignore_errors=True)
    if not best["nonempty"]:
        print("SELECTED A (no candidate has a non-empty source diff; /workspace unchanged)")
        return
    if best["name"] == "A":
        print("SELECTED A (already in /workspace; %d changed lines in %s)" % (best["nlines"], ",".join(best["paths"])))
        return
    # apply the winner to /workspace
    a = cands[0]
    keep_dir = tempfile.mkdtemp(prefix="sel_keep_")
    pa, pw = os.path.join(keep_dir, "a.patch"), os.path.join(keep_dir, "w.patch")
    with open(pa, "wb") as fh:
        fh.write(a["raw"])
    with open(pw, "wb") as fh:
        fh.write(best["text"].encode("utf-8", "surrogateescape"))
    ref = base_ref(ws)

    def reset_and_apply(patch_file, expect_text, label):
        git(ws, "reset", "--hard", "-q", ref)
        git(ws, "clean", "-fdq")
        if os.path.getsize(patch_file) == 0:
            return True
        rc, _, err = git(ws, "apply", "--binary", "--whitespace=nowarn", patch_file)
        if rc != 0:
            print("apply %s failed: %s" % (label, err.strip()[:200]))
            return False
        again = filtered(raw_diff(ws))[0]
        same = again == expect_text
        if not same:
            print("re-diff of /workspace differs from %s (%d vs %d chars)" % (label, len(again), len(expect_text)))
        return same
    ok = reset_and_apply(pw, best["text"], "winner " + best["name"])
    if ok:
        print("SELECTED %s VERIFIED (/workspace diff == hand %s diff; %d changed lines in %s; %.1fs)" % (
            best["name"], best["name"], best["nlines"], ",".join(best["paths"]), time.time() - T0))
    else:
        reset_and_apply(pa, filtered(a["raw"])[0], "hand A backup")
        print("SELECTED A (apply of %s failed: A restored)" % best["name"])
    shutil.rmtree(keep_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass
    try:
        main(sys.argv[1:])
    except Exception as e:  # noqa: BLE001
        import traceback
        print("select_attempt.py internal error: %s: %s" % (type(e).__name__, str(e)[:300]))
        print(traceback.format_exc()[-600:])
        print("SELECTED A (selector error; /workspace left as is)")
