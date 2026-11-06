#!/usr/bin/env python3
"""Leak scanner for the Gemma-4 developer-agent release set (stdlib only).

Purpose: refuse to publish anything that embeds Kaggle Competition Data (rule 2.4.b) or private material.

Two modes
---------
build   Build a blocklist from competition-derived files that exist LOCALLY in the author's repo
        (no download, no network). The blocklist is written to a path that must stay OUTSIDE the
        published tree (default release/_private/blocklist.json). Long strings are stored as 64-bit hashes
        only, but the list of instance ids is kept in clear (they are needed for reporting).
scan    Walk a directory (the release set) and count, per file and per category, every blocklist hit,
        plus generic secret / path / private-reference patterns. Exit status 1 when any LEAK-class hit exists.

Categories (LEAK class unless noted)
  instance_id      competition instance id (fastapi_NNNNN, rich_NNNN, requests_NNNN, httpx_NNNN)
  commit_sha       base commit of a competition snapshot (from the official graph file names)
  gold_or_test     a distinctive (>= 50 chars) line that occurs in competition test output, an agent patch against a
                   competition snapshot, or a gold/test patch
  graph_text       a distinctive (>= 60 chars) source line of the official code graphs (competition repos' code)
  graph_node_id    dotted node id of the official graph (>= 3 components, first component = competition repo module)
  wheel_name       name of a wheel shipped in the competition wheelhouse
  harness_text     a distinctive line of the harness / wheelhouse source (diff of the swegemma / adk wheels)
  secret           key/token/private-key patterns, e-mail addresses
  local_path       absolute personal path (C:/Users/..., OneDrive, user name)
  INFO (not a leak): private_ref (private Kaggle resource slug), harness_readme_mention, repo_name_mention

Usage
  python release/scan_competition_leaks.py build --out release/_private/blocklist.json
  python release/scan_competition_leaks.py scan release/pkg --blocklist release/_private/blocklist.json --json out.json
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ID_RE = re.compile(r"\b(?:fastapi|rich|requests|httpx)_\d{3,6}\b")
SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")
DOTTED_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){2,}")
WHEEL_RE = re.compile(r"[A-Za-z0-9_.]+-\d[\w.]*-(?:py\d|cp\d|pp\d)[\w.]*(?:-[\w.]+)*\.whl")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
SECRET_RES = {
    "kaggle_key": re.compile(r"KGAT_[0-9a-f]{20,}|\"key\"\s*:\s*\"[0-9a-f]{32}\""),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "openai_like": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer": re.compile(r"[Bb]earer\s+[A-Za-z0-9._-]{30,}"),
}
EMAIL_ALLOW = re.compile(r"^n\+@|noreply@(anthropic\.com|github\.com)|@users\.noreply\.github\.com|^git@|@example\.(com|org)")
PATH_RE = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_.~-]+|OneDrive|/home/[a-z0-9_]+/", re.I)
PRIVATE_REF_RE = re.compile(r"[A-Za-z0-9_-]+/gemma4-[A-Za-z0-9_.-]+|<kaggle-owner>/[A-Za-z0-9_.-]+")
README_RE = re.compile(r"HARNESS_README")
REPO_MENTION_RE = re.compile(r"\b(?:fastapi|tiangolo|textualize|encode/httpx|psf/requests)\b", re.I)

TEXT_EXT_SKIP = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tgz", ".whl", ".npz", ".npy", ".pyc", ".skill"}
MIN_LINE = 50
MIN_GRAPH_LINE = 60
LEAK = {"instance_id", "commit_sha", "gold_or_test", "graph_text", "graph_node_id", "wheel_name", "harness_text",
        "secret", "local_path"}


def h64(s):
    return struct.unpack("<Q", hashlib.sha1(s.encode("utf8", "replace")).digest()[:8])[0]


def norm_line(line):
    s = line.rstrip("\r\n")
    if s[:1] in "+-" and s[:3] not in ("+++", "---"):
        s = s[1:]
    return " ".join(s.split())


def trivial(line):
    """Lines too generic to be evidence of a copy (boilerplate, separators, imports, blank-ish)."""
    if len(set(line)) < 8:
        return True
    low = line.lower()
    return low.startswith(("import ", "from ", "#", "def test_", "assert ")) and len(line) < 70


# --------------------------------------------------------------------------------------- build
STRING_KEYS = ("agent_patch", "test_output_tail", "gold_detail", "empty_detail", "patch", "test_patch",
               "problem_statement", "hints_text", "diff", "gold_patch", "text")


def iter_jsonl(path):
    with open(path, encoding="utf8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    yield json.loads(ln)
                except Exception:
                    continue


def walk_strings(o, depth=0):
    if depth > 6:
        return
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from walk_strings(v, depth + 1)
    elif isinstance(o, list):
        for v in o:
            yield from walk_strings(v, depth + 1)


def add_lines(text, bucket, minlen):
    for ln in text.splitlines():
        s = norm_line(ln)
        if len(s) >= minlen and not trivial(s):
            bucket.add(h64(s))


def extra_terms():
    """Host user name (and its first dash-separated part) plus $LEAK_EXTRA_TERMS (comma list: handles, e-mail locals)."""
    import getpass
    u = getpass.getuser()
    t = {u, u.split("-")[0]}
    t |= {x.strip() for x in os.environ.get("LEAK_EXTRA_TERMS", "").split(",") if x.strip()}
    return sorted(x for x in t if len(x) >= 4)


def build(out, repo=REPO_ROOT, verbose=True, comp_dir=None):
    ids, shas, wheels = set(), set(), set()
    gold_lines, graph_lines, node_hashes, harness_lines = set(), set(), set(), set()
    sources = Counter()

    def rel(p):
        return os.path.relpath(p, repo).replace("\\", "/")

    # 1) instance ids + test output + agent patches from per-task competition-derived rows
    jsonl_srcs = [
        "research/gold_control/gold_control_tasks.jsonl",
        "research/lb-eval/lb_eval_tasks.jsonl",
    ]
    for d in ("datasets/gemma4-lb-eval-ckpt", "datasets/gemma4-paper-ckpt", "research/lb-eval-v4"):
        p = os.path.join(repo, d)
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.endswith(".jsonl"):
                    jsonl_srcs.append(f"{d}/{fn}")
    for src in jsonl_srcs:
        p = os.path.join(repo, src)
        if not os.path.exists(p):
            continue
        n = 0
        for row in iter_jsonl(p):
            if not isinstance(row, dict):
                continue
            iid = row.get("instance_id")
            if isinstance(iid, str) and ID_RE.fullmatch(iid):
                ids.add(iid)
            for k in STRING_KEYS:
                v = row.get(k)
                if isinstance(v, str) and len(v) >= MIN_LINE:
                    add_lines(v, gold_lines, MIN_LINE)
            n += 1
        sources[src] = n
    # sandbox / probe output captured by our own runs (grader-image facts, competition code in tracebacks)
    import glob as _glob
    for pat in ("research/top10/ab/mock_evidence/*/*.json", "research/top10/ab/session1/*.json",
                "research/gold_control/env_probe.json", "research/top10/fourhands/preflight_*.json"):
        for p in _glob.glob(os.path.join(repo, pat)):
            try:
                d = json.load(open(p, encoding="utf8"))
            except Exception:
                continue
            for sv in walk_strings(d):
                for w in WHEEL_RE.findall(sv):
                    wheels.add(w)
                # our own run logs (status/session_report) are not competition text: names only
                if len(sv) >= MIN_LINE and os.path.basename(p) in ("mock_checks.json", "env_probe.json") or \
                        len(sv) >= MIN_LINE and os.path.basename(p).startswith("preflight_"):
                    add_lines(sv, gold_lines, MIN_LINE)
            sources[rel(p)] = 1
    # ids listed in json files
    for src in ("research/gold_control/solvable_ids.json", "research/top10/wheelhouse-0211/strip_check_gold_patches.json"):
        p = os.path.join(repo, src)
        if os.path.exists(p):
            txt = open(p, encoding="utf8").read()
            for m in ID_RE.findall(txt):
                ids.add(m)
            sources[src] = 1

    # 2) official graphs (REP re-keyed copies of the competition graphs, local): node ids, source lines, commit shas
    gdir = os.path.join(repo, "kernels/gemma4-build-rep-graphs/output/graphs_rep")
    if os.path.isdir(gdir):
        node_re = re.compile(r'\{"id": "((?:[^"\\]|\\.)*)", "name": "(?:[^"\\]|\\.)*", "text": "((?:[^"\\]|\\.)*)"\}')
        nfiles = 0
        for fn in sorted(os.listdir(gdir)):
            m = re.match(r"(fastapi|rich|requests|httpx)_([0-9a-f]{40})\.json$", fn)
            if not m:
                continue
            shas.add(m.group(2))
            nfiles += 1
            # streaming-free but simple: the files are <= ~6 MB each
            try:
                g = json.load(open(os.path.join(gdir, fn), encoding="utf8"))
            except Exception:
                continue
            for nd in g.get("nodes", []):
                nid = nd.get("id", "")
                if nid.count(".") >= 2:
                    node_hashes.add(h64(nid))
                t = nd.get("text", "")
                if t:
                    add_lines(t, graph_lines, MIN_GRAPH_LINE)
        sources["graphs_rep (files)"] = nfiles

    # 3) wheelhouse wheel names + harness source lines (diffs of the competition wheels)
    wdir = os.path.join(repo, "research/top10/wheelhouse-0211")
    if os.path.isdir(wdir):
        for fn in sorted(os.listdir(wdir)):
            p = os.path.join(wdir, fn)
            if not os.path.isfile(p):
                continue
            txt = open(p, encoding="utf8", errors="replace").read()
            for w in WHEEL_RE.findall(txt):
                wheels.add(w)
            if fn.endswith(".diff") and not fn.startswith("lb-eval"):
                add_lines(txt, harness_lines, MIN_GRAPH_LINE)
                sources[f"wheelhouse-0211/{fn}"] = txt.count("\n")
    # wheel names listed elsewhere (manifests of what the sandbox received)
    for src in ("research/envfix-feasibility/vendor_manifest.json",):
        p = os.path.join(repo, src)
        if os.path.exists(p):
            for w in WHEEL_RE.findall(open(p, encoding="utf8", errors="replace").read()):
                wheels.add(w)

    # 4) optional: an existing local copy of the competition files (tasks.jsonl, HARNESS_README.md, wheels/*.whl).
    #    Read-only; only hashes of long lines and short identifiers are kept in the blocklist.
    if comp_dir and os.path.isdir(comp_dir):
        tp = os.path.join(comp_dir, "tasks.jsonl")
        if os.path.exists(tp):
            n = 0
            for row in iter_jsonl(tp):
                iid = row.get("instance_id")
                if isinstance(iid, str):
                    ids.add(iid)
                bc = row.get("base_commit")
                if isinstance(bc, str) and len(bc) == 40:
                    shas.add(bc)
                for k in ("patch", "test_patch", "problem_statement", "hints_text"):
                    v = row.get(k)
                    if isinstance(v, str):
                        add_lines(v, gold_lines, MIN_LINE)
                n += 1
            sources["comp_dir/tasks.jsonl"] = n
        hp = os.path.join(comp_dir, "HARNESS_README.md")
        if os.path.exists(hp):
            add_lines(open(hp, encoding="utf8", errors="replace").read(), harness_lines, 40)
            sources["comp_dir/HARNESS_README.md"] = 1
        wd = os.path.join(comp_dir, "wheels")
        if os.path.isdir(wd):
            import zipfile
            nw = 0
            for fn in sorted(os.listdir(wd)):
                if fn.endswith(".whl"):
                    wheels.add(fn)
                    try:
                        with zipfile.ZipFile(os.path.join(wd, fn)) as z:
                            for nm in z.namelist():
                                if nm.endswith((".py", ".md", ".yaml", ".yml", ".txt", ".json")):
                                    add_lines(z.read(nm).decode("utf8", "replace"), harness_lines, MIN_GRAPH_LINE)
                        nw += 1
                    except Exception:
                        pass
            sources["comp_dir/wheels"] = nw

    bl = {
        "format": 1,
        "note": "PRIVATE: derived from competition data. Never publish this file.",
        "sources": dict(sources),
        "ids": sorted(ids),
        "commit_shas": sorted(shas),
        "wheels": sorted(wheels),
        "gold_or_test": sorted(gold_lines),
        "graph_text": sorted(graph_lines),
        "graph_node_id": sorted(node_hashes),
        "harness_text": sorted(harness_lines),
        "extra_terms": extra_terms(),  # author identity / host names (private; never published)
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(bl, f)
    if verbose:
        print("blocklist:", {k: (len(v) if isinstance(v, list) else v) for k, v in bl.items() if k != "note"})
    return bl


# --------------------------------------------------------------------------------------- scan
class Blocklist:
    def __init__(self, path):
        d = json.load(open(path))
        self.ids = set(d["ids"])
        self.shas = set(d["commit_shas"])
        self.wheels = set(d["wheels"])
        self.gold = set(d["gold_or_test"])
        self.gtext = set(d["graph_text"])
        self.nodes = set(d["graph_node_id"])
        self.harness = set(d["harness_text"])
        self.extra = [t.lower() for t in d.get("extra_terms", [])]
        self.stats = {k: len(v) for k, v in d.items() if isinstance(v, list)}


def scan_text(text, bl, self_exempt=False):
    c = Counter()
    ex = defaultdict(list)

    def hit(cat, example):
        c[cat] += 1
        if len(ex[cat]) < 3:
            ex[cat].append(example[:100])

    for m in ID_RE.finditer(text):
        hit("instance_id", m.group(0))
    for m in SHA_RE.finditer(text):
        if m.group(0) in bl.shas:
            hit("commit_sha", m.group(0))
    for m in WHEEL_RE.finditer(text):
        if m.group(0) in bl.wheels:
            hit("wheel_name", m.group(0))
    for ln in text.splitlines():
        s = norm_line(ln)
        if len(s) >= MIN_LINE and not trivial(s):
            hv = h64(s)
            if hv in bl.gold:
                hit("gold_or_test", s)
            if len(s) >= MIN_GRAPH_LINE and hv in bl.gtext:
                hit("graph_text", s)
            if len(s) >= MIN_GRAPH_LINE and hv in bl.harness:
                hit("harness_text", s)
    for m in DOTTED_RE.finditer(text):
        tok = m.group(0)
        if h64(tok) in bl.nodes:
            hit("graph_node_id", tok)
    for name, rx in SECRET_RES.items():
        for m in rx.finditer(text):
            hit("secret", name)
    for m in EMAIL_RE.finditer(text):
        if not EMAIL_ALLOW.search(m.group(0)):
            hit("secret", "email:" + m.group(0))
    for m in PATH_RE.finditer(text):
        hit("local_path", m.group(0))
    low = text.lower()
    for term in getattr(bl, "extra", []):
        n = low.count(term)
        if n:
            c["local_path"] += n
            hit_ex = ex["local_path"]
            if len(hit_ex) < 3:
                hit_ex.append("term:" + term[:3] + "...")
    for m in PRIVATE_REF_RE.finditer(text):
        hit("private_ref", m.group(0))
    n = len(README_RE.findall(text))
    if n:
        c["harness_readme_mention"] += n
    n = len(REPO_MENTION_RE.findall(text))
    if n:
        c["repo_name_mention"] += n
    return c, ex


def scan_dir(root, bl, skip_dirs=("_private", ".git", "__pycache__", ".pytest_cache"), max_mb=50):
    report = {"root": root, "files": {}, "totals": Counter(), "oversize": [], "binary_skipped": []}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip_dirs]
        for fn in sorted(fns):
            if fn == "scan_competition_leaks.py":
                continue  # the scanner embeds its own patterns and self-test strings
            p = os.path.join(dp, fn)
            rp = os.path.relpath(p, root).replace("\\", "/")
            size = os.path.getsize(p)
            if size > max_mb * 1024 * 1024:
                report["oversize"].append(rp)
            ext = os.path.splitext(fn)[1].lower()
            if ext in TEXT_EXT_SKIP:
                report["binary_skipped"].append(rp)
                # names of binaries can still leak
                c, ex = scan_text(fn, bl)
                if any(k in LEAK for k in c):
                    report["files"][rp] = {"counts": dict(c), "examples": dict(ex), "size": size}
                    for k, v in c.items():
                        report["totals"][k] += v
                continue
            try:
                text = open(p, encoding="utf8", errors="replace").read()
            except Exception as e:
                report["binary_skipped"].append(rp)
                continue
            c, ex = scan_text(text + "\n" + fn, bl)
            if ext in (".json", ".jsonl"):
                # JSON escapes newlines: also scan the decoded string values line by line
                dec = []
                try:
                    objs = [json.loads(text)] if ext == ".json" else [json.loads(l) for l in text.splitlines() if l.strip()]
                    for o in objs:
                        dec.extend(walk_strings(o))
                except Exception:
                    pass
                if dec:
                    c2, ex2 = scan_text("\n".join(dec), bl)
                    for k in ("gold_or_test", "graph_text", "harness_text"):
                        if c2.get(k):
                            c[k] += c2[k]
                            ex[k] = (ex.get(k) or []) + ex2.get(k, [])
            if c:
                report["files"][rp] = {"counts": dict(c), "examples": dict(ex), "size": size}
                for k, v in c.items():
                    report["totals"][k] += v
    report["totals"] = dict(report["totals"])
    report["leak_files"] = sorted(f for f, d in report["files"].items() if any(k in LEAK for k in d["counts"]))
    report["n_leak_hits"] = sum(v for k, v in report["totals"].items() if k in LEAK)
    return report


def selftest():
    bl = Blocklist.__new__(Blocklist)
    W = "sweg" + "emma-0.2.11-py3-none-any.whl"
    FA = "fast" + "api"
    bl.ids, bl.shas, bl.wheels = set(), {"a" * 40}, {W}
    line = "    return JSONResponse({'detail': exc.detail}, status_code=exc.status_code, headers=headers)"
    bl.gold, bl.gtext, bl.nodes, bl.harness = {h64(norm_line(line))}, set(), {h64(FA + ".routing.APIRoute")}, set()
    txt = f"x fastapi_12345 {'a' * 40} swegemma-0.2.11-py3-none-any.whl\n+{line}\nfastapi.routing.APIRoute\nC:\\Users\\Foo\\x me@x.org"
    c, _ = scan_text(txt, bl)
    need = {"instance_id": 1, "commit_sha": 1, "wheel_name": 1, "gold_or_test": 1, "graph_node_id": 1, "local_path": 1, "secret": 1}
    for k, v in need.items():
        assert c.get(k, 0) >= v, (k, dict(c))
    clean, _ = scan_text("def add(a, b):\n    return a + b\n", bl)
    assert not any(k in LEAK for k in clean), dict(clean)
    print("selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=os.path.join(REPO_ROOT, "release", "_private", "blocklist.json"))
    b.add_argument("--repo", default=REPO_ROOT)
    b.add_argument("--competition-dir", help="optional local folder with tasks.jsonl, HARNESS_README.md, wheels/")
    s = sub.add_parser("scan")
    s.add_argument("root")
    s.add_argument("--blocklist", default=os.path.join(REPO_ROOT, "release", "_private", "blocklist.json"))
    s.add_argument("--json")
    s.add_argument("--quiet", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
        return 0
    if a.cmd == "build":
        build(a.out, a.repo, comp_dir=a.competition_dir)
        return 0
    bl = Blocklist(a.blocklist)
    rep = scan_dir(a.root, bl)
    rep["blocklist_sizes"] = bl.stats
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1, sort_keys=True)
    if not a.quiet:
        for f, d in sorted(rep["files"].items()):
            print(f"{f}: {d['counts']}")
        print("TOTALS", rep["totals"])
        print("OVERSIZE", rep["oversize"])
        print("LEAK FILES", len(rep["leak_files"]), "LEAK HITS", rep["n_leak_hits"])
    return 1 if rep["n_leak_hits"] or rep["oversize"] else 0


if __name__ == "__main__":
    sys.exit(main())
