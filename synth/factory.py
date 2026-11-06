"""Synthetic SWE task factory (runs in the Kaggle CPU kernel gemma4-synth-factory, internet ON).

Per repo: clone the latest release tag, check licence + contamination filter, download pure-python test
dependency wheels, build a base snapshot, set up an emulated grader sandbox, run every test file (baseline +
per-test line coverage via sys.monitoring), then screen AST mutations: keep a mutation iff the module still
compiles/collects and 1..8 previously passing tests fail quickly. A minority of 2-hunk tasks pairs two kept
mutations. Every kept bug becomes a candidate task (gold patch, snapshot patch with the failing tests removed,
test_patch re-adding them, generated problem statement). Validation with the official verify_task is a
separate kernel (synth/validate.py).

Output (/kaggle/working/candidates): tasks.jsonl, meta.jsonl, snapshots/base_snapshots/*.tgz,
snapshots/patches/*.patch, wheels/*.whl, repos.json, factory_summary.json, screen_log.jsonl.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import shutil
import tarfile
import threading
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from . import statements
from .common import (SCRATCH, WORK, Sandbox, build_merged_data_dir, comp_data_dir, install_harness, lb_module,
                     log, make_cfg, minutes, norm, sh)
from .filter import COMPETITION_WHEEL_PACKAGES, deny_reason, licence_ok
from .mutators import Mutation, apply_edits, find_mutations
from .patches import find_test_def, make_patch, parse_nodeid, remove_tests
from .repos import REPOS, SKIP_DEPS

DEFAULTS = {
    "REPOS": None,                 # None = all of synth.repos.REPOS; else list of owner/name
    "MAX_FILE_SECONDS": 20.0,      # a test file is usable only if it passes whole in <= this (grader emulation)
    "MAX_FAIL": 8,                 # failing tests per task
    "MAX_COVER": 250,              # skip sites covered by more tests than this (too broad)
    "MAX_FAIL_DUR": 8.0,           # total runtime of the failing tests (s)
    "ALARM": 10,                   # per-test timeout during screening (s)
    "TARGET_SINGLES": 60,          # accepted single mutations per repo (before pairing)
    "PAIR_FRACTION": 0.18,         # share of a repo's tasks built as 2-hunk pairs
    "MAX_PER_FUNC": 2,             # accepted singles per function
    "MAX_OP_SHARE": 0.22,          # cap of one operator among a repo's accepted singles
    "REPO_MINUTES": 55,            # screening budget per repo
    "MAX_ATTEMPTS": 900,
    "WORKERS": 4,                  # repos screened in parallel
    "SESSION_MINUTES": 640,        # stop screening new repos after this
    "SEED": 20261006,
}
STATE_LOCK = threading.Lock()


# ============================================================================= repo preparation
def latest_tag(url):
    from packaging.version import InvalidVersion, Version
    r = sh(["git", "ls-remote", "--tags", "--refs", url], timeout=120)
    best = None
    for line in r.stdout.splitlines():
        ref = line.split("refs/tags/")[-1].strip()
        m = re.match(r"^(?:v|version-|release-|[A-Za-z_-]+-)?(\d+(?:\.\d+)*(?:\.post\d+)?)$", ref)
        if not m:
            continue
        try:
            v = Version(m.group(1))
        except InvalidVersion:
            continue
        if v.is_prerelease or v.is_devrelease:
            continue
        if best is None or v > best[0]:
            best = (v, ref)
    return best[1] if best else None


def github_meta(repo):
    import urllib.request
    try:
        with urllib.request.urlopen(f"https://api.github.com/repos/{repo}", timeout=20) as r:
            d = json.load(r)
        lic = (d.get("license") or {}).get("spdx_id")
        return {"spdx": lic, "archived": d.get("archived"), "fork": d.get("fork"), "stars": d.get("stargazers_count"),
                "full_name": d.get("full_name")}
    except Exception as e:
        return {"spdx": None, "error": repr(e)[:200]}


def licence_from_files(repo_dir: Path):
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.rst", "LICENSE.md", "COPYING", "LICENCE"):
        p = repo_dir / name
        if p.exists():
            t = p.read_text(errors="ignore")[:4000]
            if "Apache License" in t and "Version 2.0" in t:
                return "Apache-2.0"
            if "Permission is hereby granted, free of charge" in t:
                return "MIT"
            if "Redistribution and use in source and binary forms" in t:
                return "BSD-3-Clause" if ("Neither the name" in t or "names of its contributors" in t or
                                          "name of the copyright holder" in t) else "BSD-2-Clause"
            if "Permission to use, copy, modify, and/or distribute" in t:
                return "ISC"
            if "PYTHON SOFTWARE FOUNDATION" in t.upper():
                return "PSF-2.0"
            return "unknown"
    return None


def package_paths(repo_dir: Path, packages):
    """Relative paths of the import packages/modules inside the repo (src/ or flat layout)."""
    out = []
    for pkg in packages:
        for cand in (f"src/{pkg}", pkg, f"src/{pkg}.py", f"{pkg}.py", f"lib/{pkg}"):
            p = repo_dir / cand
            if p.exists():
                out.append(cand)
                break
    return out


def source_files(repo_dir: Path, pkg_paths):
    files = []
    for pp in pkg_paths:
        p = repo_dir / pp
        cands = [p] if p.is_file() else sorted(p.rglob("*.py"))
        for f in cands:
            rel = f.relative_to(repo_dir).as_posix()
            parts = rel.split("/")
            if any(x in ("tests", "test", "testing", "_vendor", "vendor", "vendored", "benchmarks") for x in parts):
                continue
            if parts[-1] in ("_version.py", "__main__.py", "conftest.py", "version.py", "__about__.py") or \
                    parts[-1].startswith("test_"):
                continue
            files.append(rel)
    return files


def test_files(repo_dir: Path):
    out = []
    for f in sorted(repo_dir.rglob("*.py")):
        rel = f.relative_to(repo_dir).as_posix()
        parts = rel.split("/")
        if parts[0] in ("docs", "doc", "examples", "benchmarks", "bench", "build", "dist", ".git", "tools",
                        "scripts") or any(p.startswith(".") for p in parts):
            continue
        name = parts[-1]
        if (name.startswith("test_") or name.endswith("_test.py")) and name != "test_.py":
            out.append(rel)
        elif name in ("tests.py", "test.py") and (len(parts) <= 2 or parts[-2] in ("tests", "test")):
            out.append(rel)
    return out


def has_c_ext(repo_dir: Path, pkg_paths):
    for pp in pkg_paths:
        p = repo_dir / pp
        if p.is_dir() and (any(p.rglob("*.c")) or any(p.rglob("*.pyx")) or any(p.rglob("*.rs"))):
            return True
    return False


def declared_deps(repo_dir: Path):
    """Runtime deps + test extras/groups from pyproject.toml / setup.cfg / requirement files."""
    import tomllib
    deps = []
    pp = repo_dir / "pyproject.toml"
    if pp.exists():
        try:
            d = tomllib.loads(pp.read_text(encoding="utf-8"))
        except Exception:
            d = {}
        proj = d.get("project", {})
        deps += proj.get("dependencies", [])
        for g, v in (proj.get("optional-dependencies") or {}).items():
            if re.search(r"test", g, re.I):
                deps += v
        for g, v in (d.get("dependency-groups") or {}).items():
            if re.search(r"test", g, re.I):
                deps += [x for x in v if isinstance(x, str)]
        poetry = d.get("tool", {}).get("poetry", {})
        for k in (poetry.get("dependencies") or {}):
            if k.lower() != "python":
                deps.append(k)
        for g, v in (poetry.get("group") or {}).items():
            if re.search(r"test|dev", g, re.I):
                deps += list((v.get("dependencies") or {}).keys())
    for rf in list(repo_dir.glob("requirements*test*.txt")) + list(repo_dir.glob("requirements/test*.txt")) + \
            list(repo_dir.glob("test*requirements*.txt")) + list(repo_dir.glob("tests/requirements*.txt")):
        for line in rf.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")):
                deps.append(line)
    names = []
    for dspec in deps:
        if ";" in dspec:
            marker = dspec.split(";", 1)[1]
            if re.search(r"python_version\s*<\s*['\"]3\.(\d+)", marker) and \
                    int(re.search(r"python_version\s*<\s*['\"]3\.(\d+)", marker).group(1)) <= 13:
                continue
            if "platform_python_implementation" in marker and "PyPy" in marker and "==" in marker:
                continue
        m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", dspec)
        if m:
            names.append(m.group(1))
    return names


def download_pure_wheels(names, dest: Path, exclude_norm: set):
    """pip download each requirement (with deps) as pure-python wheels only. -> (got, failed)"""
    dest.mkdir(parents=True, exist_ok=True)
    got, failed = set(), {}
    queue = list(dict.fromkeys(names))
    seen = set()
    while queue:
        n = queue.pop(0)
        if norm(n) in seen or norm(n) in exclude_norm:
            continue
        seen.add(norm(n))
        tmp = dest / f"_dl_{norm(n)}"
        shutil.rmtree(tmp, ignore_errors=True)
        r = sh(["python3", "-m", "pip", "download", "-q", "--no-deps", "--only-binary=:all:", "--platform", "any",
                "--python-version", "3.13", "--implementation", "py", "--abi", "none", "-d", str(tmp), n], timeout=300)
        whl = sorted(tmp.glob("*.whl")) if tmp.exists() else []
        if r.returncode != 0 or not whl:
            failed[n] = (r.stderr or r.stdout)[-300:]
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        for w in whl:
            shutil.move(str(w), dest / w.name)
            got.add(w.name)
            # transitive requirements
            import zipfile
            with zipfile.ZipFile(dest / w.name) as zf:
                meta = next((x for x in zf.namelist() if x.endswith(".dist-info/METADATA")), None)
                text = zf.read(meta).decode("utf-8", "ignore") if meta else ""
            for line in text.splitlines():
                if line.startswith("Requires-Dist:"):
                    raw = line.split(":", 1)[1].strip()
                    if "extra ==" in raw:
                        continue
                    if ";" in raw and re.search(r"python_version\s*<\s*['\"]3\.(\d+)", raw):
                        mv = int(re.search(r"python_version\s*<\s*['\"]3\.(\d+)", raw).group(1))
                        if mv <= 13:
                            continue
                    if ";" in raw and re.search(r"sys_platform\s*==\s*['\"]win32", raw):
                        continue
                    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", raw)
                    if m:
                        queue.append(m.group(1))
        shutil.rmtree(tmp, ignore_errors=True)
    return sorted(got), failed


def build_base_snapshot(repo_dir: Path, out_tgz: Path):
    """Pristine tree + a fresh empty git repo (no commits: the harness's baseline commit becomes the only one,
    so `git log -p` cannot reveal the injected bug or the removed tests)."""
    shutil.rmtree(repo_dir / ".git", ignore_errors=True)
    sh(["git", "init", "-q", "-b", "main"], cwd=repo_dir, check=True)
    # repo-local identity: the harness's baseline commit must work even where the sandbox env has no global git
    # config (the emulation's sanitized env); without it there is no HEAD and agent diffs come out empty
    sh(["git", "config", "user.email", "agent@eval"], cwd=repo_dir, check=True)
    sh(["git", "config", "user.name", "Agent"], cwd=repo_dir, check=True)
    shutil.rmtree(repo_dir / ".git" / "hooks", ignore_errors=True)
    out_tgz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tgz, "w:gz", compresslevel=9) as tf:
        tf.add(repo_dir, arcname=".", filter=lambda ti: None if ("__pycache__" in ti.name or
                                                               ti.name.endswith(".pyc")) else ti)
    return out_tgz.stat().st_size


def prepare_repo(spec, all_packages, root: Path):
    repo = spec["repo"]
    info = {"repo": repo, "packages": spec["packages"]}
    for pkg in spec["packages"]:
        why = deny_reason(repo, pkg)
        if why:
            info["skip"] = f"filter: {why}"
            return info
    url = f"https://github.com/{repo}.git"
    gm = github_meta(repo)
    info["github"] = gm
    tag = spec.get("tag") or latest_tag(url)
    if not tag:
        info["skip"] = "no release tag"
        return info
    slug = re.sub(r"[^a-z0-9]", "", spec["packages"][0].lower())
    d = root / "repos" / slug
    shutil.rmtree(d, ignore_errors=True)
    d.parent.mkdir(parents=True, exist_ok=True)
    r = sh(["git", "clone", "-q", "--depth", "1", "--branch", tag, url, str(d)], timeout=600)
    if r.returncode != 0:
        info["skip"] = f"clone failed: {r.stderr[-300:]}"
        return info
    info.update({"tag": tag, "slug": slug, "url": url,
                 "commit": sh(["git", "rev-parse", "HEAD"], cwd=d).stdout.strip(),
                 "commit_date": sh(["git", "log", "-1", "--format=%cI"], cwd=d).stdout.strip()})
    lic_file = licence_from_files(d)
    spdx = gm.get("spdx") if gm.get("spdx") not in (None, "NOASSERTION") else lic_file
    info["licence"] = spdx
    info["licence_file_guess"] = lic_file
    if not licence_ok(spdx):
        info["skip"] = f"licence {spdx!r} not permissive"
        return info
    pk = package_paths(d, spec["packages"])
    if len(pk) != len(spec["packages"]):
        info["skip"] = f"package dirs not found: {spec['packages']} -> {pk}"
        return info
    info["pkg_paths"] = pk
    if has_c_ext(d, pk):
        info["skip"] = "C/Cython/Rust sources in package"
        return info
    info["source_files"] = source_files(d, pk)
    info["test_files"] = test_files(d)
    if not info["test_files"]:
        info["skip"] = "no test files"
        return info
    deps = [x for x in declared_deps(d) + list(spec.get("test_deps", []))
            if norm(x) not in {norm(s) for s in SKIP_DEPS} and not norm(x).startswith("pytest_cov")
            and not norm(x).startswith("types_")]
    info["declared_deps"] = sorted(set(deps))
    info["repo_dir"] = str(d)
    return info


# ============================================================================= screening
class RepoRun:
    def __init__(self, info, mgr, cfg, base_tgz, plug_dir, cfgd, rng):
        self.info, self.mgr, self.cfg, self.base_tgz, self.plug, self.C, self.rng = \
            info, mgr, cfg, base_tgz, plug_dir, cfgd, rng
        self.slug = info["slug"]
        self.repo_dir = Path(info["repo_dir"])
        self.out = SCRATCH / "out" / self.slug
        self.out.mkdir(parents=True, exist_ok=True)
        self.n_runs = 0
        self.log_rows = []

    def start(self):
        self.sb = Sandbox(self.mgr, self.cfg, self.info["repo"], self.base_tgz)
        for d in self.sb.ws.rglob("__pycache__"):
            shutil.rmtree(d, ignore_errors=True)
        self.ws_real = os.path.realpath(self.sb.ws)

    def run(self, targets, trace=(), alarm=0, timeout=300):
        self.n_runs += 1
        return self.sb.pytest(targets, self.out / f"run_{self.n_runs}.json", self.plug, trace=trace, alarm=alarm,
                              timeout=timeout)

    def rel(self, p):
        rp = os.path.realpath(p)
        return os.path.relpath(rp, self.ws_real).replace(os.sep, "/") if rp.startswith(self.ws_real) else None

    # ------------------------------------------------------------------ baseline
    def baseline(self):
        C = self.C
        self.file_ok, self.file_info = {}, {}
        self.passing = set()
        self.nonpass_names = set()             # short names of skipped/xfailed tests (junit <skipped>)
        self.cover = defaultdict(set)          # (relfile, line) -> nodeids
        trace = tuple(self.info["pkg_paths"])
        for tf in self.info["test_files"]:
            r, wall, data = self.run([tf], trace=trace, alarm=60, timeout=600)
            st = {"exit": r.exit_code, "wall": round(wall, 1)}
            if data is None:
                st["why"] = "no plugin output"
            else:
                tests = data["tests"]
                st["n"] = len(tests)
                st["failed"] = sum(1 for v in tests.values() if v["outcome"] == "failed")
                st["collect_errors"] = len(data["collect_errors"])
            ok = data is not None and r.exit_code == 0 and st.get("n", 0) > 0 and not st.get("failed")
            if ok:
                # timing without tracing; must also pass a second time (flakiness)
                r2, wall2, data2 = self.run([tf], alarm=60, timeout=600)
                st["wall2"] = round(wall2, 1)
                ok = r2.exit_code == 0 and wall2 <= C["MAX_FILE_SECONDS"] and data2 is not None and \
                    {k for k, v in data2["tests"].items() if v["outcome"] == "passed"} == \
                    {k for k, v in data["tests"].items() if v["outcome"] == "passed"}
                if not ok:
                    st["why"] = "slow or flaky"
            else:
                st.setdefault("why", "fails in grader emulation")
                if data is not None:
                    errs = [e["longrepr"][-300:] for e in data["collect_errors"]][:2]
                    fails = [f"{k}: {v.get('exc_type')}" for k, v in data["tests"].items()
                             if v["outcome"] == "failed"][:3]
                    st["detail"] = errs + fails
                else:
                    st["detail"] = [(r.stdout or "")[-400:], (r.stderr or "")[-300:]]
            self.file_ok[tf] = ok
            self.file_info[tf] = st
            if ok:
                for nid, v in data["tests"].items():
                    if v["outcome"] == "passed":
                        self.passing.add(nid)
                    else:
                        self.nonpass_names.add(parse_nodeid(nid)[2])
                for nid, cov in data["cover"].items():
                    if nid not in self.passing:
                        continue
                    for f, lines in cov.items():
                        rf = self.rel(f)
                        if rf:
                            for ln in lines:
                                self.cover[(rf, ln)].add(nid)
        usable = [k for k, v in self.file_ok.items() if v]
        log(f"[{self.slug}] baseline: {len(usable)}/{len(self.file_ok)} test files usable, "
            f"{len(self.passing)} passing tests, {len(self.cover)} covered lines")
        return usable

    # ------------------------------------------------------------------ mutations
    def candidates(self):
        out = []
        self.src_text = {}
        for rel in self.info["source_files"]:
            p = self.repo_dir / rel
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            self.src_text[rel] = text
            for m in find_mutations(text, rel):
                lines = range(m.fn_first + 1, m.fn_last + 1) if m.op == "default_value" else m.lines()
                cov = set()
                for ln in lines:
                    cov |= self.cover.get((rel, ln), set())
                if 1 <= len(cov) <= self.C["MAX_COVER"]:
                    m.cover = sorted(cov)
                    out.append(m)
        return out

    def order(self, cands):
        by_op = defaultdict(list)
        for m in cands:
            by_op[m.op].append(m)
        for op in by_op:
            self.rng.shuffle(by_op[op])
            by_op[op].sort(key=lambda m: (not m.public, not m.has_doc))
            # interleave preference tiers but keep randomness inside a tier
        ops = sorted(by_op)
        self.rng.shuffle(ops)
        order = []
        while any(by_op.values()):
            for op in ops:
                if by_op[op]:
                    order.append(by_op[op].pop(0))
        return order

    def apply(self, muts):
        """Write the mutated files into the sandbox workspace. -> {file: (orig, mutated)}"""
        changes = {}
        by_file = defaultdict(list)
        for m in muts:
            by_file[m.file].extend(m.edits)
        for f, edits in by_file.items():
            orig = self.src_text[f]
            new = apply_edits(orig.encode("utf-8"), edits).decode("utf-8")
            changes[f] = (orig, new)
            (self.sb.ws / f).write_bytes(new.encode("utf-8"))
        return changes

    def restore(self, changes):
        for f, (orig, _) in changes.items():
            (self.sb.ws / f).write_bytes(orig.encode("utf-8"))

    def screen(self, muts, cover):
        """Apply muts, run the covering tests. -> (status, failures dict nodeid->record, wall)"""
        changes = self.apply(muts)
        try:
            for f, (_, new) in changes.items():
                try:
                    compile(new, f, "exec")
                except SyntaxError:
                    return "syntax", {}, 0.0
            targets = sorted(cover)
            if len(targets) > 150:
                targets = sorted({parse_nodeid(n)[0] for n in cover})
            r, wall, data = self.run(targets, alarm=self.C["ALARM"], timeout=60)
            if data is None:
                return "timeout_or_crash", {}, wall
            if data["collect_errors"]:
                return "collect_error", {}, wall
            fails = {k: v for k, v in data["tests"].items() if v["outcome"] == "failed" and k in self.passing}
            if any(v.get("exc_type") == "SynthTimeout" for v in fails.values()):
                return "test_timeout", fails, wall
            if not fails:
                return "survived", {}, wall
            if len(fails) > self.C["MAX_FAIL"]:
                return "too_many_failures", fails, wall
            if sum(v.get("dur", 0) for v in fails.values()) > self.C["MAX_FAIL_DUR"]:
                return "slow_failures", fails, wall
            for nid, v in fails.items():
                path, classes, name, _ = parse_nodeid(nid)
                # verify_task's junit check matches required names by prefix (name_, name[, nameN): a skipped or
                # xfailed test with such a name would make the gold patch fail verification
                if any(s == name or s.startswith(name + "_") or s.startswith(name + "[") or
                       (s.startswith(name) and s[len(name):].isdigit()) for s in self.nonpass_names):
                    return "f2p_name_collides_with_skipped", fails, wall
                df = v.get("def_file")
                if df and self.rel(df) != path:
                    return "test_defined_elsewhere", fails, wall
                try:
                    import ast
                    fn, _ = find_test_def(ast.parse((self.repo_dir / path).read_text(encoding="utf-8")), classes, name)
                except Exception:
                    fn = None
                if fn is None:
                    return "test_not_removable", fails, wall
            return "ok", fails, wall
        finally:
            self.restore(changes)

    def screen_all(self, t_end):
        C = self.C
        cands = self.order(self.candidates())
        log(f"[{self.slug}] {len(cands)} covered mutation sites; ops={dict(Counter(m.op for m in cands))}")
        accepted, per_func, per_op = [], Counter(), Counter()
        stats = Counter()
        attempts = 0
        for m in cands:
            if len(accepted) >= C["TARGET_SINGLES"] or attempts >= C["MAX_ATTEMPTS"] or time.time() > t_end:
                break
            fkey = f"{m.file}::{m.func}"
            if per_func[fkey] >= C["MAX_PER_FUNC"]:
                continue
            if per_op[m.op] >= max(3, int(C["MAX_OP_SHARE"] * C["TARGET_SINGLES"])):
                continue
            attempts += 1
            status, fails, wall = self.screen([m], m.cover)
            stats[status] += 1
            self.log_rows.append({"repo": self.slug, "op": m.op, "file": m.file, "func": m.func, "line": m.lineno,
                                  "status": status, "n_fail": len(fails), "wall": round(wall, 2),
                                  "n_cover": len(m.cover)})
            if status == "ok":
                accepted.append((m, fails))
                per_func[fkey] += 1
                per_op[m.op] += 1
        log(f"[{self.slug}] screened {attempts}: {dict(stats)} -> {len(accepted)} accepted")
        self.stats = dict(stats)
        return accepted

    def pairs(self, accepted, n_pairs):
        """2-hunk tasks from accepted singles (same file preferred half the time)."""
        pool = list(range(len(accepted)))
        self.rng.shuffle(pool)
        used, out = set(), []
        tries = 0
        while len(out) < n_pairs and tries < n_pairs * 6:
            tries += 1
            free = [i for i in pool if i not in used]
            if len(free) < 2:
                break
            i = free[0]
            mi, fi = accepted[i]
            same = self.rng.random() < 0.5
            cands = [j for j in free[1:] if accepted[j][0].func != mi.func and
                     ((accepted[j][0].file == mi.file) == same) and
                     not (accepted[j][0].file == mi.file and not (accepted[j][0].end_lineno < mi.lineno or
                                                                   mi.end_lineno < accepted[j][0].lineno)) and
                     len(set(fi) | set(accepted[j][1])) <= self.C["MAX_FAIL"]]
            if not cands:
                cands = [j for j in free[1:] if accepted[j][0].func != mi.func and
                         not (accepted[j][0].file == mi.file and not (accepted[j][0].end_lineno < mi.lineno or
                                                                       mi.end_lineno < accepted[j][0].lineno)) and
                         len(set(fi) | set(accepted[j][1])) <= self.C["MAX_FAIL"]]
            if not cands:
                used.add(i)
                continue
            j = cands[0]
            mj, fj = accepted[j]
            status, fails, wall = self.screen([mi, mj], set(mi.cover) | set(mj.cover))
            self.log_rows.append({"repo": self.slug, "op": f"{mi.op}+{mj.op}", "file": f"{mi.file}|{mj.file}",
                                  "func": f"{mi.func}|{mj.func}", "status": "pair_" + status, "n_fail": len(fails),
                                  "wall": round(wall, 2)})
            if status == "ok" and set(fails) == set(fi) | set(fj):
                out.append(((mi, mj), fails, [sorted(fi), sorted(set(fj) - set(fi))]))
                used |= {i, j}
            else:
                used.add(j if self.rng.random() < 0.5 else i)
        used_idx = set()
        for (mi, mj), _, _ in out:
            for k, (m, _) in enumerate(accepted):
                if m is mi or m is mj:
                    used_idx.add(k)
        singles = [accepted[k] for k in range(len(accepted)) if k not in used_idx]
        return singles, out


# ============================================================================= task packaging
def build_task(iid, rr: RepoRun, muts, fails, info, groups=None):
    test_changes, removed_all = {}, []
    by_file = defaultdict(list)
    for nid in fails:
        path, classes, name, _ = parse_nodeid(nid)
        by_file[path].append((classes, name))
    for path, targets in by_file.items():
        orig = (rr.repo_dir / path).read_text(encoding="utf-8")
        new, removed = remove_tests(orig, targets)
        test_changes[path] = (orig, new)
        removed_all += [f"{path}::{'::'.join(c + (n,))}" for c, n in removed]
    src_changes = {}
    by_src = defaultdict(list)
    for m in muts:
        by_src[m.file].extend(m.edits)
    for f, edits in by_src.items():
        orig = rr.src_text[f]
        src_changes[f] = (orig, apply_edits(orig.encode("utf-8"), edits).decode("utf-8"))
    start_patch = make_patch({**{f: (o, n) for f, (o, n) in src_changes.items()},
                              **{f: (o, n) for f, (o, n) in test_changes.items()}})
    gold = make_patch({f: (n, o) for f, (o, n) in src_changes.items()})
    test_patch = make_patch({f: (n, o) for f, (o, n) in test_changes.items()})
    failures = []
    for nid, rec in sorted(fails.items()):
        path = parse_nodeid(nid)[0]
        failures.append(dict(rec, nodeid=nid, test_src=test_changes[path][0]))
    roots = sorted({rr.ws_real, str(rr.sb.ws), rr.ws_real.replace("/private", "")}, key=len, reverse=True)
    stmt, style, sinfo = statements.generate(iid, failures, info["packages"], sorted(src_changes), roots=roots,
                                             groups=groups)
    task = {
        "instance_id": iid, "repo": info["repo"], "base_commit": info["commit"], "problem_statement": stmt,
        "patch": gold, "test_patch": test_patch, "hints_text": "", "created_at": info["commit_date"],
        "FAIL_TO_PASS": sorted(fails), "PASS_TO_PASS": [], "version": info["tag"],
        "environment_setup_commit": info["commit"],
    }
    meta = {
        "instance_id": iid, "repo": info["repo"], "slug": info["slug"], "licence": info["licence"],
        "kind": "single" if len(muts) == 1 else "pair", "ops": [m.op for m in muts],
        "files": sorted({m.file for m in muts}), "funcs": [f"{m.file}::{m.func}" for m in muts],
        "lines": [[m.lineno, m.end_lineno] for m in muts], "details": [m.detail for m in muts],
        "public": all(m.public for m in muts), "has_doc": any(m.has_doc for m in muts),
        "n_f2p": len(fails), "removed_tests": removed_all, "style": style, "symbol": sinfo.get("symbol"),
        "symbols": sinfo.get("symbols"), "fail_exc_types": sorted({v.get("exc_type") or "?" for v in fails.values()}),
        "fail_dur_s": round(sum(v.get("dur", 0) for v in fails.values()), 3),
        "gold_lines_changed": sum(1 for l in gold.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))),
        "test_files": sorted(by_file),
    }
    return task, meta, start_patch


# ============================================================================= main
def main(CONFIG):
    C = dict(DEFAULTS)
    C.update(CONFIG or {})
    rng_master = random.Random(C["SEED"])
    out = WORK / "candidates"
    shutil.rmtree(out, ignore_errors=True)
    (out / "snapshots" / "base_snapshots").mkdir(parents=True)
    (out / "snapshots" / "patches").mkdir(parents=True)
    (out / "wheels").mkdir(parents=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    plug = SCRATCH / "plug"
    plug.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__).with_name("synth_plug.py"), plug / "synth_plug.py")
    summary = {"config": C, "repos": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    rc = install_harness()
    log(f"harness installed (pip rc={rc})")

    specs = [s for s in REPOS if C["REPOS"] is None or s["repo"] in C["REPOS"]]
    all_pkgs = {norm(p) for s in REPOS for p in s["packages"]}
    # ---------------- phase 1: repos + wheels
    infos = []
    for spec in specs:
        try:
            info = prepare_repo(spec, all_pkgs, SCRATCH)
        except Exception:
            info = {"repo": spec["repo"], "skip": "EXCEPTION " + traceback.format_exc()[-800:]}
        log(f"repo {spec['repo']}: " + (f"SKIP {info['skip']}" if info.get("skip") else
                                        f"{info['tag']} {info['commit'][:10]} {info['licence']} "
                                        f"src={len(info['source_files'])} tests={len(info['test_files'])} "
                                        f"deps={info['declared_deps']}"))
        infos.append(info)
    exclude = set(COMPETITION_WHEEL_PACKAGES) | all_pkgs
    wheel_dir = out / "wheels"
    all_deps = sorted({d for i in infos if not i.get("skip") for d in i["declared_deps"]})
    got, failed = download_pure_wheels(all_deps, wheel_dir, exclude)
    # drop older duplicates (the harness injects the highest version per name anyway)
    from packaging.version import Version
    best = {}
    for w in sorted(wheel_dir.glob("*.whl")):
        n, v = w.name.split("-")[:2]
        k = norm(n)
        try:
            vv = Version(v)
        except Exception:
            vv = Version("0")
        if k in best and best[k][0] >= vv:
            w.unlink()
            continue
        if k in best:
            best[k][1].unlink()
        best[k] = (vv, w)
    wheels = sorted(w.name for w in wheel_dir.glob("*.whl"))
    summary["wheels"] = {"files": wheels, "failed": failed,
                         "total_bytes": sum(w.stat().st_size for w in wheel_dir.glob("*.whl"))}
    log(f"wheels: {len(wheels)} pure wheels ({summary['wheels']['total_bytes'] / 1e6:.2f} MB): {wheels}; "
        f"not available as pure wheels: {sorted(failed)}")

    # ---------------- phase 2: snapshots, emulation
    snaps = out / "snapshots"
    from swegemma.deduplication import get_base_snapshot_filename
    for info in infos:
        if info.get("skip"):
            continue
        tgz = snaps / "base_snapshots" / get_base_snapshot_filename(info["repo"])
        work_copy = SCRATCH / "snapcopy" / info["slug"]
        shutil.rmtree(work_copy, ignore_errors=True)
        shutil.copytree(info["repo_dir"], work_copy, symlinks=True)
        info["snapshot_bytes"] = build_base_snapshot(work_copy, tgz)
        info["base_snapshot"] = str(tgz)
        shutil.rmtree(work_copy, ignore_errors=True)
        shutil.rmtree(Path(info["repo_dir"]) / ".git", ignore_errors=True)
    comp = comp_data_dir()
    data_dir = build_merged_data_dir(SCRATCH / "data", comp, None, snaps, wheel_dir)
    LB = lb_module()
    LB.setup_emulation(data_dir)
    from .common import host_import_ok, make_manager
    host_before = {m: host_import_ok(m) for m in ("mistune", "pyparsing", "tabulate", "markdown")}
    mgr = make_manager(LB)
    cfg = make_cfg(data_dir, SCRATCH / "results")
    log(f"emulation ready: {LB.ENV.get('sandbox_python')} image={len(LB.ENV['image']['image_pkgs'])} pkgs")

    # ---------------- phase 3: per repo baseline + screening (parallel over repos)
    results = {}

    def do_repo(info):
        if minutes() > C["SESSION_MINUTES"]:
            return info["repo"], {"skip": "session budget"}
        rr = RepoRun(info, mgr, cfg, Path(info["base_snapshot"]), plug, C,
                     random.Random(f"{C['SEED']}:{info['repo']}"))
        try:
            t = time.time()
            rr.start()
            info["setup_s"] = round(time.time() - t, 1)
            usable = rr.baseline()
            info["file_status"] = rr.file_info
            info["usable_test_files"] = usable
            if not usable:
                return info["repo"], {"skip": "no usable test files", "rr": rr}
            t_end = time.time() + 60 * C["REPO_MINUTES"]
            acc = rr.screen_all(t_end)
            n_pairs = int(round(len(acc) * C["PAIR_FRACTION"] / (1 + C["PAIR_FRACTION"])))
            singles, prs = rr.pairs(acc, n_pairs)
            log(f"[{rr.slug}] {len(singles)} singles + {len(prs)} pairs")
            return info["repo"], {"rr": rr, "singles": singles, "pairs": prs}
        except Exception:
            log(f"[{info['repo']}] EXCEPTION {traceback.format_exc()[-1500:]}")
            return info["repo"], {"skip": "exception", "rr": rr}

    todo = [i for i in infos if not i.get("skip")]
    with cf.ThreadPoolExecutor(max_workers=C["WORKERS"]) as ex:
        for repo, res in ex.map(do_repo, todo):
            results[repo] = res

    # ---------------- phase 4: package candidates
    tasks_f = open(out / "tasks.jsonl", "w", encoding="utf-8")
    meta_f = open(out / "meta.jsonl", "w", encoding="utf-8")
    screen_f = open(out / "screen_log.jsonl", "w", encoding="utf-8")
    n_tasks = Counter()
    for info in infos:
        res = results.get(info["repo"], {})
        rr = res.get("rr")
        if rr is not None:
            for row in rr.log_rows:
                screen_f.write(json.dumps(row) + "\n")
        if "singles" not in res:
            continue
        items = [((m,), f, None) for m, f in res["singles"]] + [((a, b), f, g) for (a, b), f, g in res["pairs"]]
        items.sort(key=lambda x: (len(x[0]), x[0][0].file, x[0][0].lineno, x[0][0].op))
        k = 0
        for muts, fails, groups in items:
            iid = f"{info['slug']}_s{k:04d}"
            try:
                task, meta, start_patch = build_task(iid, rr, muts, fails, info, groups)
            except Exception as e:
                log(f"[{info['slug']}] build_task failed for {[m.key for m in muts]}: {e!r}")
                continue
            (snaps / "patches" / f"{iid}.patch").write_text(start_patch, encoding="utf-8")
            tasks_f.write(json.dumps(task) + "\n")
            meta_f.write(json.dumps(meta) + "\n")
            n_tasks[info["repo"]] += 1
            k += 1
        rr.sb.stop()
    for f in (tasks_f, meta_f, screen_f):
        f.close()
    for info in infos:
        res = results.get(info["repo"], {})
        rr = res.get("rr")
        summary["repos"][info["repo"]] = {
            k: info.get(k) for k in ("skip", "tag", "commit", "commit_date", "licence", "licence_file_guess",
                                     "github", "packages", "pkg_paths", "declared_deps", "snapshot_bytes",
                                     "setup_s", "usable_test_files", "file_status")}
        summary["repos"][info["repo"]]["n_candidates"] = n_tasks.get(info["repo"], 0)
        if res.get("skip"):
            summary["repos"][info["repo"]]["skip"] = res["skip"]
        if rr is not None and hasattr(rr, "stats"):
            summary["repos"][info["repo"]]["screen_stats"] = rr.stats
    summary["n_candidates"] = sum(n_tasks.values())
    summary["minutes"] = round(minutes(), 1)
    # unused base snapshots (skipped repos) are removed
    used = {get_base_snapshot_filename(r) for r in n_tasks}
    for f in (snaps / "base_snapshots").glob("*.tgz"):
        if f.name not in used:
            f.unlink()
    (out / "repos.json").write_text(json.dumps(summary["repos"], indent=1, default=str))
    (out / "factory_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (WORK / "factory_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    summary["pip_shims"] = type(mgr).shims
    summary["host_imports_before_after"] = {m: (v, host_import_ok(m)) for m, v in host_before.items()}
    (WORK / "factory_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"pip shims written: {type(mgr).shims}; host imports before/after: {summary['host_imports_before_after']}")
    log(f"DONE {summary['n_candidates']} candidates: {dict(n_tasks)}")
