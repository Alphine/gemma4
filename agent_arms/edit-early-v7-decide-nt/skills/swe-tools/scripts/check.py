"""check.py - compile + import a module, then run ONE existing test file, compact summary.

Usage (run_skill_script): args {"path": "pkg/module.py", "test": "tests/test_module.py"}
- path: the source file you edited (or a dotted module). Compiles it and imports it (timeout 20 s).
- test (optional): test file(s) or node ids, comma separated; "auto" picks the best existing test file.
  Runs `python -m pytest <test> -x -q` with a 60 s timeout. Never runs a whole suite.
- k (optional): pytest -k expression.
Prints: compile result, import result, passed/failed counts, the first failure's assertion lines.
"""
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locate import find_root, list_py, module_of, parse_args, rank_tests  # noqa: E402

MAX_OUT = 2600


def run(cmd, cwd, timeout):
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("COLUMNS", "120")
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout, env=env)
        out = (r.stdout or b"").decode("utf-8", "replace") + (r.stderr or b"").decode("utf-8", "replace")
        return r.returncode, out, False, time.time() - t
    except subprocess.TimeoutExpired as e:
        out = e.stdout or b""
        out = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
        return None, out, True, time.time() - t


def resolve(root, files, target):
    t = re.sub(r"^(\./|/workspace/)", "", target.strip().replace("\\", "/"))
    if t.startswith(root.rstrip("/") + "/"):
        t = t[len(root.rstrip("/")) + 1:]
    if t.endswith(".py"):
        return t, module_of(t)
    mod = t
    p = mod.replace(".", "/")
    for cand in (p + ".py", p + "/__init__.py", "src/" + p + ".py", "src/" + p + "/__init__.py"):
        if cand in files:
            return cand, mod
    return None, mod


def summarize_pytest(out):
    lines = out.splitlines()
    res = []
    summ = ""
    for l in reversed(lines):
        if re.search(r"\b\d+ (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b", l) or "no tests ran" in l:
            summ = l.strip("= ").strip()
            break
    res.append("tests: " + (summ or "(no pytest summary line)"))
    failed = [l for l in lines if l.startswith(("FAILED ", "ERROR "))]
    if failed:
        res.append("first failing: " + failed[0][:300])
    # first failure block: location + E lines
    elines, locs = [], []
    for l in lines:
        if l.startswith("E ") and len(elines) < 14:
            elines.append(l[:200])
        elif re.match(r"^[\w/.\-]+\.py:\d+: \w", l) and len(locs) < 4:
            locs.append(l[:200])
    if locs:
        res.append("at: " + " | ".join(locs))
    if elines:
        res.append("\n".join(elines))
    if not failed and not elines and (not summ or "error" in summ or "no tests ran" in summ):
        tail = [l for l in lines if l.strip()][-15:]
        res.append("\n".join(x[:200] for x in tail))
    return "\n".join(res)


def main(argv):
    opts, pos = parse_args(argv)
    g = opts.get
    target = g("path") or g("module") or g("file") or g("source") or (pos[0] if pos else "")
    test = g("test") or g("tests") or g("test_file") or (pos[1] if len(pos) > 1 else "")
    kexpr = g("k") or g("keyword")
    test_timeout = float(g("timeout") or 60)
    root = find_root(opts)
    files = list_py(root)
    out = []
    src = None
    if target:
        src, mod = resolve(root, files, target)
        if src:
            try:
                with open(os.path.join(root, src), "rb") as fh:
                    code = fh.read()
                compile(code, src, "exec", dont_inherit=True)
                out.append("compile %s: OK" % src)
            except SyntaxError as e:
                out.append("compile %s: SyntaxError line %s: %s | %s" % (src, e.lineno, e.msg, (e.text or "").strip()[:120]))
                print("\n".join(out))
                return
            except Exception as e:
                out.append("compile %s: %s %s" % (src, type(e).__name__, e))
        else:
            out.append("source %s not found as a file; importing it as a module" % target)
        if mod and not src_is_script(src):
            code = ("import importlib,sys\nm=importlib.import_module(%r)\nprint('import OK:', getattr(m,'__file__',''))" % mod)
            rc, o, to, dt = run([sys.executable, "-c", code], root, 20)
            if to:
                out.append("import %s: TIMEOUT after 20 s (import-time hang?)" % mod)
            elif rc == 0:
                f = (o.strip().splitlines() or [""])[-1].replace("import OK:", "").strip()
                where = "" if (not f or os.path.abspath(f).startswith(root)) else "  WARNING: imported from outside the repo (%s)" % f
                out.append("import %s: OK%s" % (mod, where))
            else:
                tb = [l for l in o.strip().splitlines() if l.strip()]
                out.append("import %s: FAILED\n%s" % (mod, "\n".join(x[:200] for x in tb[-8:])))
    if test:
        tests = [t.strip() for t in re.split(r"[,\s]+", test) if t.strip()][:3]
        if tests == ["auto"]:
            if not src:
                out.append("test=auto needs a source path")
                tests = []
            else:
                r = rank_tests(root, files, src, limit=1)
                tests = [r[0][1]] if r else []
                out.append("auto test file: %s" % (tests[0] if tests else "(none found)"))
        bad = [t for t in tests if t in (".", "tests", "tests/", "test", "test/") or os.path.isdir(os.path.join(root, t))]
        if bad:
            out.append("refusing to run a whole directory (%s): pass a test file" % ", ".join(bad))
            tests = [t for t in tests if t not in bad]
        if tests:
            cmd = [sys.executable, "-m", "pytest", *tests, "-x", "-q", "-rfE", "--tb=short", "-p", "no:cacheprovider",
                   "--color=no"]
            if kexpr:
                cmd += ["-k", kexpr]
            rc, o, to, dt = run(cmd, root, test_timeout)
            if to:
                out.append("pytest %s: TIMEOUT after %d s (hang/infinite loop?). Partial output:" % (" ".join(tests), test_timeout))
                out.append("\n".join(l[:200] for l in o.strip().splitlines()[-10:]))
            else:
                out.append("pytest %s (%.1f s, exit %s):" % (" ".join(tests), dt, rc))
                out.append(summarize_pytest(o))
    if not out:
        out.append('usage: args {"path": "pkg/module.py", "test": "tests/test_module.py"}')
    s = "\n".join(out)
    print(s[:MAX_OUT] + ("\n...[truncated]" if len(s) > MAX_OUT else ""))


def src_is_script(src):
    """Files outside a package (scripts/, top-level .py) are not imported as modules."""
    if not src:
        return False
    parts = src.split("/")
    return parts[0] in ("scripts", "script", "tools", "bin", "docs", "docs_src", "examples") or src in ("setup.py", "noxfile.py")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    try:
        main(sys.argv[1:])
    except Exception as e:
        print("check.py internal error: %s: %s" % (type(e).__name__, str(e)[:300]))
