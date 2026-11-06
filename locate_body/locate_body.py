"""locate.py - rank the definitions an issue is most likely about.

Usage (run_skill_script): args {"issue": "<issue title + key lines>"}   (or a list ["<text>"])
Optional: "k" (default 8), "tests": "true" to also search test files.
Prints the top definitions as `path:line  kind  Class.name` with 2 context lines, then the
existing test files of the top 3 files. Pure stdlib, read-only.
"""
import math
import os
import re
import subprocess
import sys

BOOL_FLAGS = {"tests", "all"}
N_MORE = 12            # extra candidates printed compactly after the detailed top-k (no context lines)
# each heuristic can be switched off for ablations: SWE_LOCATE_OFF=docstr,pkgname,...
OFF = set(x.strip() for x in os.environ.get("SWE_LOCATE_OFF", "").split(",") if x.strip())


MAX_OUT = 2900 if "v5" in OFF else 3300


def on(flag):
    return flag not in OFF


def use(feature):
    """a v6 signal runs only when it is not switched off AND has a positive weight (zero-weight signals cost nothing)"""
    return feature not in OFF and WEIGHTS.get(feature, 0.0) > 0


# ----------------------------------------------------------------------------- shared helpers
def parse_args(argv, bool_flags=()):
    opts, pos, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--":
            pos.extend(argv[i + 1:])
            break
        if a.startswith("--") and len(a) > 2:
            k = a[2:]
            if "=" in k and re.match(r"^[A-Za-z_-]+=", k):
                k, v = k.split("=", 1)
            elif k.lower().replace("-", "_") in bool_flags and (i + 1 >= len(argv) or argv[i + 1].lower() not in ("true", "false", "1", "0", "yes", "no")):
                v = "true"
            elif i + 1 < len(argv):
                v = argv[i + 1]
                i += 1
            else:
                v = "true"
            opts[k.lower().replace("-", "_")] = v
        elif re.match(r"^-[A-Za-z]$", a) and i + 1 < len(argv):
            opts[a[1:]] = argv[i + 1]
            i += 1
        else:
            pos.append(a)
        i += 1
    return opts, pos


def truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def find_root(opts):
    cands = [opts.get("root"), os.environ.get("SWE_TOOLS_ROOT")]
    f = sys._getframe()
    while f is not None:   # run_skill_script's wrapper keeps the original cwd in a local
        if "_orig_cwd" in f.f_locals:
            cands.append(f.f_locals["_orig_cwd"])
        f = f.f_back
    cands += [os.environ.get("PWD"), "/workspace", os.getcwd()]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, ".git")):
            return os.path.abspath(c)
    for c in cands:
        if c and os.path.isdir(c):
            return os.path.abspath(c)
    return os.getcwd()


SKIP_DIRS = {".git", ".tox", ".venv", "venv", "env", "node_modules", "build", "dist", "site-packages",
             "__pycache__", ".mypy_cache", ".pytest_cache", ".eggs"}


def list_py(root):
    try:
        r = subprocess.run(["git", "-C", root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.py"],
                           capture_output=True, timeout=15)
        files = [p for p in r.stdout.decode("utf-8", "replace").split("\0") if p.endswith(".py")]
        if r.returncode == 0 and files:
            return sorted(set(p for p in files if not os.path.basename(p).startswith(".adk_exec_")))
    except Exception:
        pass
    out = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.endswith(".egg-info")]
        for f in fs:
            if f.endswith(".py") and not f.startswith(".adk_exec_"):
                out.append(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    return sorted(out)


def is_test_path(p):
    parts = p.lower().split("/")
    name = parts[-1]
    return (any(x in ("tests", "test", "testing") for x in parts[:-1]) or name.startswith("test_")
            or name.endswith("_test.py") or name == "conftest.py")


NON_SOURCE = ("docs", "doc", "docs_src", "examples", "example", "benchmarks", "benchmark", ".github", "site")


def is_non_source(p):
    return p.split("/")[0].lower() in NON_SOURCE


def read_text(path):
    try:
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", "replace")
    except Exception:
        return ""


def module_of(path):
    p = path[:-3] if path.endswith(".py") else path
    parts = p.split("/")
    if parts and parts[0] in ("src", "lib"):
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


# ----------------------------------------------------------------------------- tests_for core (shared with tests_for.py)
def rank_tests(root, files, src, symbol=None, limit=5):
    """-> [(score, test_path, reasons)] for the source file src."""
    tests = [p for p in files if is_test_path(p) and not is_non_source(p)
             and re.match(r"^test_.*\.py$|^.*_test\.py$", os.path.basename(p))]
    mod = module_of(src)
    stem = os.path.basename(src)[:-3] if src.endswith(".py") else os.path.basename(src)
    if stem == "__init__":
        stem = mod.split(".")[-1] if mod else stem
    pkg, _, last = mod.rpartition(".")
    text = read_text(os.path.join(root, src))
    names = set(re.findall(r"^(?:async\s+)?def\s+([A-Za-z]\w{3,})|^class\s+([A-Za-z]\w{2,})", text, re.M))
    names = {a or b for a, b in names}
    names = {n for n in names if not n.startswith("test")}
    imp_res = []
    if mod:
        imp_res.append(re.compile(r"^\s*(?:from\s+%s\s+import|import\s+%s\b)" % (re.escape(mod), re.escape(mod)), re.M))
    if pkg and last:
        imp_res.append(re.compile(r"^\s*from\s+%s\s+import\s+(?:\([^)]*\b%s\b|[^\n]*\b%s\b)" % (re.escape(pkg), re.escape(last), re.escape(last)), re.M))
    sym_re = re.compile(r"\b%s\b" % re.escape(symbol)) if symbol else None
    out = []
    sdirs = src.lower().split("/")[:-1]
    parent = sdirs[-1] if sdirs and not (len(sdirs) == 1 or (len(sdirs) == 2 and sdirs[0] in ("src", "lib"))) else ""
    for t in tests:
        tname = os.path.basename(t)[:-3].lower()
        core = re.sub(r"^test_|_test$", "", tname)
        score, why = 0.0, []
        tparts = [x for x in re.split(r"[_\W]+", tname) if x and x != "test"]
        if tname in ("test_" + stem.lower(), stem.lower() + "_test"):
            score += 10
            why.append("name")
        elif stem.lower() in tparts or ("_".join(tparts) and stem.lower() == "_".join(tparts)):
            score += 6
            why.append("name~")
        elif len(stem) >= 4 and stem.lower() in tname:
            score += 3
            why.append("name~")
        elif on("mirror") and core and len(core) >= 4 and stem.lower().startswith(core):
            score += 3   # tutorial002_py310.py <- test_tutorial002.py (variant suffixes of one example)
            why.append("name~")
        if on("mirror") and parent and len(parent) >= 3 and any(x in (parent, "test_" + parent) for x in t.lower().split("/")[:-1]):
            score += 4   # tests/<...>/test_<dir>/ mirrors <...>/<dir>/
            why.append("dir " + parent)
        body = read_text(os.path.join(root, t))
        if any(r.search(body) for r in imp_res):
            score += 8
            why.append("imports " + (mod or stem))
        hits = 0
        for n in names:
            if n in body and re.search(r"\b%s\b" % re.escape(n), body):
                hits += 1
        if hits:
            score += min(6.0, 1.0 * hits)
            why.append("%d symbols" % hits)
        if sym_re is not None:
            c = len(sym_re.findall(body))
            if c:
                score += 6 + min(6, c)
                why.append("%s x%d" % (symbol, c))
        if score > 0:
            out.append((score, t, why))
    out.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
    return out[:limit]


# ----------------------------------------------------------------------------- term extraction
STOP = set("""
a an the and or but if then else when while for to of in on at by with from as is are was were be been being it its
this that these those not no yes can cant could should would will may might must do does did done has have had we i
you he she they them our your my me us there here what which who whom how why where so than too very just also only
some any all each more most other such into out up down over under again about after before because between both
same own now use used using uses like get gets got make makes made see seems seem etc eg ie fix fixes fixed
fixing bug bugs issue issues error errors problem problems add adds added adding support supports supported update
updates updated allow allows allowed handle handles handled handling change changes changed pull request pr
feature features documentation docs doc test tests tested testing code python versions please thanks
thank expected expect actual behavior behaviour example examples case cases work works working worked correct
correctly wrong instead still already https http github com www org one two true false none null
returns returned self def class import checklist description merge closes close closed related
want wants need needs needed able unable there's doesn't isn't don't won't it's can't i'm should've
something anything nothing thing things way ways lot bit sure well good bad better best really
however though although via per since until within without currently previously previous
following follows below above calls called calling runs running results
functions attribute objects instance arguments parameters strings lists exceptions messages warnings
breaking minor major release changelog
""".split())
# plausible identifiers but weak evidence in prose: kept at a low weight
WEAK = set("""new old type types print value values return note notes time times number set sets line lines file files
call run result output outputs input inputs option options default defaults method methods function object argument
parameter string list dict int str bool float tuple raise raises raised exception traceback message warning version
first last one change data name names""".split())
PY_KEYWORDS = set("""False None True and as assert async await break class continue def del elif else except finally
for from global if import in is lambda nonlocal not or pass raise return try while with yield self cls print""".lower().split())

URL_RE = re.compile(r"https?://\S+")
TEMPLATE_RE = re.compile(r"(?s:<!--.*?-->)|^\s*[-*] \[[ xX]\][^\n]*|^#+ *(Type of changes|Checklist|Description)\b[^\n]*", re.M)
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+")


def name_parts(name):
    out = []
    for chunk in re.split(r"[_\W]+", name):
        if not chunk:
            continue
        out += [m.lower() for m in CAMEL_RE.findall(chunk)]
    return [p for p in out if p]


def norm(name):
    return re.sub(r"[_\W]+", "", name).lower()


def stem_eq(a, b):
    if a == b:
        return True
    if len(a) < 4 or len(b) < 4:
        return False
    for x, y in ((a, b), (b, a)):
        if x == y + "s" or x == y + "es" or (x.endswith("ies") and x[:-3] + "y" == y):
            return True
    return False


def extract_terms(text, pkg_names=()):
    """-> dict term -> (weight, parts, kind). kind: code/trace/title/word/join/phrase."""
    text = text or ""
    text = TEMPLATE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    terms = {}

    def add(term, w, kind):
        term = term.strip("._")
        if not term or len(term) < 3 or term.lower() in PY_KEYWORDS:
            return
        parts = name_parts(term)
        if not parts:
            return
        key = norm(term)
        if len(key) < 3:
            return
        old = terms.get(key)
        if old is None or w > old[0]:
            terms[key] = (w, parts, kind, term)

    phrases = []
    lines = text.splitlines()
    title = next((l for l in lines if l.strip()), "")
    # traceback frames: File ".../pkg/mod.py", line N, in func
    paths = []
    for m in re.finditer(r'File "([^"]+\.py)", line (\d+), in (\w+)', text):
        paths.append(m.group(1).replace("\\", "/"))
        if m.group(3) not in ("<module>",):
            add(m.group(3), 3.0, "trace")
    # exception lines: FooError: message
    for m in re.finditer(r"\b([A-Z]\w*(?:Error|Exception|Warning))\b(?::\s*([^\n]{6,160}))?", text):
        add(m.group(1), 1.2, "code")
        if m.group(2):
            msg = m.group(2).strip().strip("'\"")
            if len(msg) >= 8:
                phrases.append(msg[:80])
    # code: fenced blocks, inline backticks
    code_chunks = re.findall(r"```[\w+-]*\n(.*?)```", text, re.S)
    inline = re.findall(r"`([^`\n]{1,120})`", re.sub(r"```.*?```", " ", text, flags=re.S))
    for chunk in inline:
        for m in re.finditer(r"[A-Za-z_][\w.]*", chunk):
            tok = m.group(0)
            pieces = [p for p in tok.split(".") if p]
            if len(pieces) > 1:
                for p in pieces[:-1]:
                    add(p, 1.5, "code")
                add(pieces[-1], 3.0, "code")
            else:
                add(tok, 3.0, "code")
    for chunk in code_chunks:
        for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", chunk):
            add(m.group(1), 1.6, "code")
        for m in re.finditer(r"\.([A-Za-z_]\w*)", chunk):
            add(m.group(1), 1.4, "code")
        for m in re.finditer(r"^\s*(?:from\s+([\w.]+)\s+)?import\s+([\w, ()]+)", chunk, re.M):
            for n in re.findall(r"\w+", m.group(2)):
                add(n, 1.4, "code")
            if m.group(1):
                for p in m.group(1).split("."):
                    add(p, 0.6, "code")
    for m in re.finditer(r"(?<![\w`])([A-Za-z_]+(?:\.[A-Za-z_]\w*)+)(?=\s*\(|\b)", re.sub(r"```.*?```", " ", text, flags=re.S)):
        pieces = m.group(1).split(".")
        if len(pieces) >= 2 and not re.match(r"^(e\.g|i\.e|etc)$", m.group(1).lower()):
            add(pieces[-1], 2.0, "code")
            for p in pieces[:-1]:
                add(p, 1.0, "code")
    # identifiers written in prose: snake_case, CamelCase, dunder
    prose = re.sub(r"```.*?```", " ", text, flags=re.S)
    for m in re.finditer(r"\b(__\w+__|[a-z]+_[a-z0-9_]+|[A-Z][a-z0-9]+[A-Z]\w*|[a-z]+[A-Z]\w*)\b", prose):
        add(m.group(1), 2.2, "code")
    for m in re.finditer(r"\b([A-Z][A-Z0-9]*_[A-Z0-9_]+)\b", prose):   # ENV_VAR / CONSTANT
        add(m.group(1), 2.5, "code")
    # quoted phrases (error messages, outputs) for literal search
    for m in re.finditer(r"[\"']([^\"'\n]{8,80})[\"']", prose):
        phrases.append(m.group(1))

    # words of the title (strong) and of the body (weak) + joins of adjacent words
    def words_of(s):
        s = re.sub(r"`[^`]*`", " ", s)
        return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_]*", s)]

    body = re.sub(r"```.*?```", " ", "\n".join(lines[1:]), flags=re.S)[:3000]
    n_body = len({w.lower() for w in words_of(body) if w.lower() not in STOP and len(w) >= 3})
    scale = min(1.0, 25.0 / max(1, n_body))   # a long body must not drown the title and the code terms
    for src_text, w_word, w_join in ((title, 1.6, 1.6), (body, 0.5 * scale, 0.8 * scale)):
        ws = words_of(src_text)
        content = [w for w in ws if w.lower() not in STOP and len(w) >= 3]
        for w in content:
            if w.lower() in pkg_names:
                continue
            add(w, w_word * (0.4 if w.lower() in WEAK else 1.0), "title" if src_text is title else "word")
        # joins of adjacent content words (stopwords break adjacency only when they are not 'for/of/the')
        seq = [w for w in ws if w.lower() not in ("the", "a", "an")]
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if a.lower() in STOP or b.lower() in STOP:
                continue
            add(a.lower() + "_" + b.lower(), w_join, "join")
            if i + 2 < len(seq) and seq[i + 2].lower() not in STOP:
                add(a.lower() + "_" + b.lower() + "_" + seq[i + 2].lower(), w_join * 0.9, "join")
    return terms, phrases, paths


# ----------------------------------------------------------------------------- definitions index
DEF_RE = re.compile(r"^([ \t]*)(?:async[ \t]+)?(def|class)[ \t]+([A-Za-z_]\w*)")
ASSIGN_RE = re.compile(r"^([A-Za-z_]\w*)[ \t]*(?::[^=\n]+)?=(?!=)")


def index_file_regex(path, text):
    defs = []
    lines = text.splitlines()
    stack = []   # (indent, name)
    in_str = None   # open triple quote: def/class lines of examples inside docstrings are not definitions
    for i, line in enumerate(lines):
        if in_str is not None:
            if line.count(in_str) % 2 == 1:
                in_str = None
            continue
        if on("docstr") and ('"""' in line or "'''" in line) and not line.lstrip().startswith("#"):
            for q in ('"""', "'''"):
                if line.count(q) % 2 == 1:
                    in_str = q
                    break
        m = DEF_RE.match(line)
        if m:
            ind = len(m.group(1).expandtabs(4))
            while stack and stack[-1][0] >= ind:
                stack.pop()
            cls = next((n for d, n, k in reversed(stack) if k == "class"), None)
            defs.append({"path": path, "line": i + 1, "kind": m.group(2), "name": m.group(3), "cls": cls,
                         "indent": ind, "qual": ".".join([n for d, n, k in stack] + [m.group(3)])})
            stack.append((ind, m.group(3), m.group(2)))
            continue
        m = ASSIGN_RE.match(line)
        if m and not line.startswith((" ", "\t")):
            stack = []
            n = m.group(1)
            if n not in ("__all__", "__version__", "logger", "log") and not n.startswith("_T"):
                defs.append({"path": path, "line": i + 1, "kind": "var", "name": n, "cls": None, "indent": 0})
        elif line and not line[0].isspace() and not line.startswith(("#", "@", ")", "]", "}")):
            stack = []
    # body extents
    for j, d in enumerate(defs):
        end = len(lines)
        for e in defs[j + 1:]:
            if e["indent"] <= d["indent"]:
                end = e["line"] - 1
                break
        if d["kind"] == "var":
            end = min(end, d["line"] + 3)
        d["end"] = end
        own = end   # lines that belong to this definition itself (a class stops at its first method)
        for e in defs[j + 1:]:
            if e["line"] > end:
                break
            if e["kind"] != "var" and e["indent"] > d["indent"]:
                own = e["line"] - 1
                break
        d["own_end"] = own
    return defs, lines


def index_file_ast(path, text):
    """Same output as index_file (regex) but from ast: exact def/class lines and end lines, no confusion by triple quotes inside
    code (the regex tracker mis-reads a line like `if s[:3] in ('\"\"\"', \"'''\"):` and then drops every later definition of the file).
    Falls back to the regex indexer when the file does not parse (python 2 code, templates, ...)."""
    import ast
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return index_file_regex(path, text)
    lines = text.splitlines()
    defs = []

    def visit(body, stack, top):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else "def"
                cls = next((n for n, k in reversed(stack) if k == "class"), None)
                d = {"path": path, "line": node.lineno, "kind": kind, "name": node.name, "cls": cls,
                     "indent": node.col_offset, "qual": ".".join([n for n, k in stack] + [node.name]),
                     "end": getattr(node, "end_lineno", None) or node.lineno}
                defs.append(d)
                inner_start = len(defs)
                visit(node.body, stack + [(node.name, kind)], False)
                nested = [x for x in defs[inner_start:] if x["kind"] != "var"]
                d["own_end"] = min([x["line"] for x in nested] or [d["end"] + 1]) - 1 if nested else d["end"]
                if d["own_end"] < d["line"]:
                    d["own_end"] = d["line"]
            elif top and isinstance(node, (ast.Assign, ast.AnnAssign)):
                tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if isinstance(tgt, ast.Name) and tgt.id not in ("__all__", "__version__", "logger", "log") and not tgt.id.startswith("_T"):
                    end = min(getattr(node, "end_lineno", node.lineno), node.lineno + 3)
                    defs.append({"path": path, "line": node.lineno, "kind": "var", "name": tgt.id, "cls": None, "indent": 0,
                                 "end": end, "own_end": end})
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)) and not isinstance(node, ast.AsyncFunctionDef):
                # definitions guarded by `if TYPE_CHECKING` / `try: import` / version checks live at the enclosing scope
                for sub in ("body", "orelse", "finalbody"):
                    visit(getattr(node, sub, []) or [], stack, False)
                for h in getattr(node, "handlers", []) or []:
                    visit(h.body, stack, False)

    visit(tree.body, [], True)
    defs.sort(key=lambda d: (d["line"], d["indent"]))
    return defs, lines


def index_file(path, text):
    """definitions of one file; ast-based unless SWE_LOCATE_OFF contains astindex (or v5), regex fallback when it does not parse"""
    if on("astindex") and on("v5"):
        return index_file_ast(path, text)
    return index_file_regex(path, text)


# ----------------------------------------------------------------------------- feature-based ranking
# Final score of a definition = sum_f WEIGHTS[f] * feature_f.  The first eight features are exactly the v5 locate.py score
# terms (WEIGHTS_V5 reproduces v5 byte for byte); the rest are the v6 signals.  Every feature can be switched off for
# ablations (SWE_LOCATE_OFF=bm25,gcall,...), and weights can be overridden for tuning (SWE_LOCATE_W="bm25=2,gcall=1").
WEIGHTS_V5 = {"name": 1.0, "stem": 1.0, "refs": 1.0, "tr": 1.0, "phr": 1.0, "tb": 1.0, "cov": 1.5, "multi": 1.0}
# v6 weights, fitted on the synthetic TRAIN split (research/locate-improve/train.py: listwise softmax, L2 prior toward v5, repo-fold CV).
# The default ("lite") keeps only the signals that measurably help: BM25 over definition chunks, direct callees of the best hits
# (gcall), definitions in the same file as the best hit (sfile), log-name score (lname).  tbf (traceback frame) is a-priori
# (too few tracebacks in the synthetic data to fit).  The FULL set also has g2 (callees of callees), gcaller (callers of the best
# hits), sclass (siblings in the best hit's class) and tcall (callees of tests that exercise the best hits): measured, no gain on
# synthetic train/held-out nor on the dev tasks, so they are off.  SWE_LOCATE_OFF=full switches them on.
WEIGHTS_LITE = {"name": 0.023, "stem": 0.0, "refs": 0.238, "tr": 0.751, "phr": 0.239, "tb": 0.73, "cov": 0.506, "multi": 0.205,
                "bm25": 1.48, "gcall": 0.498, "g2": 0.0, "gcaller": 0.0, "sclass": 0.0, "sfile": 0.453, "tcall": 0.0,
                "tbf": 2.5, "lname": 0.34}
WEIGHTS_FULL = {"name": 0.024, "stem": 0.0, "refs": 0.231, "tr": 0.742, "phr": 0.246, "tb": 0.726, "cov": 0.515, "multi": 0.195,
                "bm25": 1.454, "gcall": 0.492, "g2": 0.106, "gcaller": 0.126, "sclass": 0.344, "sfile": 0.417, "tcall": 0.121,
                "tbf": 2.5, "lname": 0.324}
WEIGHTS = dict(WEIGHTS_FULL if "full" in OFF else WEIGHTS_LITE)
if "v5" in OFF:      # SWE_LOCATE_OFF=v5: the previous (edit-early-v5) ranking, for A/B runs
    WEIGHTS = dict(WEIGHTS_V5)
    WEIGHTS.update({k: 0.0 for k in WEIGHTS_LITE if k not in WEIGHTS_V5})
STAGE1 = {"bm25": 4.0}     # stage-1 score (v5 terms + bm25) picks the seeds for the graph / class / file / test features
SEEDS = 6
NEW_FEATS = ("bm25", "gcall", "g2", "gcaller", "sclass", "sfile", "tcall")   # (tbf / lname need no repo-wide pass)
for _kv in os.environ.get("SWE_LOCATE_W", "").split(","):
    if "=" in _kv:
        _k, _v = _kv.split("=", 1)
        try:
            WEIGHTS[_k.strip()] = float(_v)
        except ValueError:
            pass

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CALL_RE = re.compile(r"(?<![\w])([A-Za-z_]\w*)\s*\(")
ATTR_CALL_RE = re.compile(r"\.([A-Za-z_]\w*)\s*\(")
DOC_RE = re.compile(r"[rRuUbB]{0,2}(\"\"\"|''')(.*?)\1", re.S)
_stem_memo = {}
_ident_memo = {}


def stem(w):
    r = _stem_memo.get(w)
    if r is None:
        r = w.lower()
        for suf in ("ing", "ed", "es", "s"):
            if len(r) > len(suf) + 3 and r.endswith(suf):
                r = r[:-len(suf)]
                break
        _stem_memo[w] = r
    return r


def ident_tokens(ident):
    """stems of the identifier's sub-tokens plus the joined identifier when it has several parts"""
    r = _ident_memo.get(ident)
    if r is None:
        parts = name_parts(ident)
        r = [stem(p) for p in parts if len(p) >= 2]
        if len(parts) > 1:
            r.append(norm(ident))
        _ident_memo[ident] = r
    return r


FRAME_RE = re.compile(r'File "([^"]+\.py)", line (\d+), in (\w+)')


def extract_frames(text):
    """traceback frames [(path, line, func)] in the order printed (innermost last)"""
    return [(m.group(1).replace("\\", "/"), int(m.group(2)), m.group(3)) for m in FRAME_RE.finditer(text or "")
            if m.group(3) != "<module>"]


class Repo(object):
    pass


def load_repo(root, want_tests=False):
    """Index every python definition under root (read-only)."""
    R = Repo()
    R.root = root
    R.files = list_py(root)
    R.want_tests = want_tests
    src_files = [p for p in R.files if want_tests or (not is_test_path(p) and not is_non_source(p))]
    if not src_files:
        src_files = R.files
    R.src_files = src_files
    R.pkg_names = {p.split("/")[1].lower() if p.startswith(("src/", "lib/")) and p.count("/") > 1 else p.split("/")[0].lower()
                   for p in src_files if "/" in p}
    R.pkg_dirs = set()   # top-level packages (dir/__init__.py or src/dir/__init__.py)
    for p in R.files:
        parts = p.split("/")
        if len(parts) == 2 and parts[1] == "__init__.py":
            R.pkg_dirs.add(parts[0])
        elif len(parts) == 3 and parts[0] in ("src", "lib") and parts[2] == "__init__.py":
            R.pkg_dirs.add(parts[0] + "/" + parts[1])
    R.index, R.file_lines, R.file_text = [], {}, {}
    for p in src_files:
        t = read_text(os.path.join(root, p))
        if not t or len(t) > 1_500_000:
            continue
        defs, lines = index_file(p, t)
        R.file_lines[p] = lines
        R.file_text[p] = t
        R.index.extend(defs)
    R.df = {}   # document frequency of name parts over definition names
    for d in R.index:
        for part in set(name_parts(d["name"])):
            R.df[part] = R.df.get(part, 0) + 1
    R.n_defs = len(R.index)
    R.file_cache = {}
    R.name_idx = {}
    for i, d in enumerate(R.index):
        d["id"] = i
        R.name_idx.setdefault(d["name"], []).append(i)
    R.built = False
    return R


def build_chunks(R):
    """per definition: weighted token counts (name x3, class x1, docstring x2, body x1), called names, inverted index;
    per test function: called names"""
    from collections import Counter
    R.built = True
    R.tf, R.dlen, R.calls, R.inv = [], [], [], {}
    for i, d in enumerate(R.index):
        lines = R.file_lines[d["path"]]
        body = lines[d["line"] - 1:d["own_end"]][:250]
        text = "\n".join(body)
        fn, fc, fd, fb = Counter(), Counter(), Counter(), Counter()
        for t in ident_tokens(d["name"]):
            fn[t] += 1.0
        if d["cls"]:
            for t in ident_tokens(d["cls"]):
                fc[t] += 1.0
        if d["kind"] != "var":
            m = DOC_RE.search(chr(10).join(body[1:14]))
            if m:
                for ident in IDENT_RE.findall(m.group(2)[:600]):
                    for t in ident_tokens(ident):
                        fd[t] += 1.0
        n = 0
        for ident in IDENT_RE.findall(text):
            n += 1
            for t in ident_tokens(ident):
                fb[t] += 1.0
        tf = (fn, fc, fd, fb)
        R.tf.append(tf)
        R.dlen.append(max(n, 5))
        calls = set(CALL_RE.findall(text)) | set(ATTR_CALL_RE.findall(text))
        calls.discard(d["name"])
        R.calls.append(calls)
        for t in set(fn) | set(fc) | set(fd) | set(fb):
            R.inv.setdefault(t, []).append(i)
    R.avglen = sum(R.dlen) / max(1, len(R.dlen))


def ensure_callers(R):
    if getattr(R, "callers_built", False):
        return
    R.callers_built = True
    R.callers = {}
    for i in range(len(R.index)):
        for j, w in callee_ids(R, i):
            R.callers.setdefault(j, []).append(i)


def ensure_tests(R):
    """test functions: name + the repo definitions they call"""
    if getattr(R, "tests_built", False):
        return
    R.tests_built = True
    R.tests = []
    for p in R.files:
        if not (is_test_path(p) and re.match(r"^test_.*\.py$|^.*_test\.py$", os.path.basename(p))) or is_non_source(p):
            continue
        t = read_text(os.path.join(R.root, p))
        if not t or len(t) > 400_000:
            continue
        defs, lines = index_file(p, t)
        for d in defs:
            if d["kind"] == "def" and d["name"].startswith("test"):
                txt = chr(10).join(lines[d["line"] - 1:d["end"]][:120])
                calls = (set(CALL_RE.findall(txt)) | set(ATTR_CALL_RE.findall(txt))) & set(R.name_idx)
                if calls:
                    R.tests.append((p, d["name"], calls))


def callee_ids(R, i):
    out = []
    for n in R.calls[i]:
        ids = R.name_idx.get(n)
        if not ids or len(ids) > 4:
            continue
        w = 1.0 / (len(ids) ** 0.5)
        for j in ids:
            if j != i:
                out.append((j, w))
            if R.index[j]["kind"] == "class":     # Foo(...) runs Foo.__init__ / __call__
                for m in ("__init__", "__call__"):
                    for q in R.name_idx.get(m, ()):
                        if R.index[q]["cls"] == R.index[j]["name"] and R.index[q]["path"] == R.index[j]["path"] and q != i:
                            out.append((q, w))
    return out


BM25P = {"n": 3.0, "c": 1.0, "d": 2.0, "b": 1.0, "k1": 1.2, "bl": 0.6}     # field weights (name, class, docstring, body), k1, length norm
for _kv in os.environ.get("SWE_LOCATE_BM25", "").split(","):
    if "=" in _kv:
        try:
            BM25P[_kv.split("=")[0].strip()] = float(_kv.split("=")[1])
        except ValueError:
            pass


def bm25_scores(R, terms):
    """BM25F-lite over definition chunks; query token weights come from extract_terms"""
    qw = {}
    for key, (w, tparts, kind, raw) in terms.items():
        if kind == "word":
            w = w * 0.6
        parts = [stem(p) for p in tparts if len(p) >= 2]
        for t in parts:
            qw[t] = max(qw.get(t, 0.0), w / (len(parts) ** 0.5))
        if len(parts) > 1:
            qw[key] = max(qw.get(key, 0.0), w)
    N = len(R.index)
    k1, b = BM25P["k1"], BM25P["bl"]
    wn, wc, wd, wb = BM25P["n"], BM25P["c"], BM25P["d"], BM25P["b"]
    out = {}
    for t, w in qw.items():
        post = R.inv.get(t)
        if not post:
            continue
        idf = math.log(1.0 + (N - len(post) + 0.5) / (len(post) + 0.5))
        for i in post:
            fn, fc, fd, fb = R.tf[i]
            tf = wn * fn.get(t, 0.0) + wc * fc.get(t, 0.0) + wd * fd.get(t, 0.0) + wb * fb.get(t, 0.0)
            out[i] = out.get(i, 0.0) + w * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * R.dlen[i] / R.avglen))
    return out, qw


def featurize(R, text):
    """-> (feats, matched, terms, phrase_hits): feats = {def id: {feature: value}} for every definition with a signal"""
    index, file_lines, file_text = R.index, R.file_lines, R.file_text
    terms, phrases, tb_paths = extract_terms(text, R.pkg_names)
    low_text = text.lower()
    pkg_dirs, df, n_defs = R.pkg_dirs, R.df, R.n_defs

    def file_factor(p):
        """code outside the importable packages (scripts/, tools/) ranks below package code."""
        top = p.split("/")[0]
        if on("outside") and pkg_dirs and "/" in p and not any(p.startswith(d + "/") for d in pkg_dirs) \
                and not is_test_path(p):
            stem_ = os.path.basename(p)[:-3].lower()
            if top.lower() not in low_text and stem_ not in low_text:
                return 0.6
        return 1.0

    def idf(part):
        return math.log(1.0 + n_defs / (1.0 + df.get(part, 0)))

    def name_score(name, matched):
        parts = name_parts(name)
        if not parts:
            return 0.0
        nn = norm(name)
        s = 0.0
        for key, (w, tparts, kind, raw) in terms.items():
            if len(tparts) == 1:
                tp = tparts[0]
                if nn == key or (len(parts) == 1 and stem_eq(parts[0], tp)):
                    s += w * idf(tp) * 2.2
                    matched.add(raw)
                elif any(stem_eq(p, tp) for p in parts):
                    s += w * idf(tp) * (1.0 if len(tp) >= 4 else 0.5)
                    matched.add(raw)
            else:
                L = len(tparts)
                if nn == key:
                    s += w * sum(idf(x) for x in tparts) * 2.0
                    matched.add(raw)
                else:
                    for i in range(len(parts) - L + 1):
                        if all(stem_eq(parts[i + j], tparts[j]) for j in range(L)):
                            s += w * sum(idf(x) for x in tparts) * 1.3
                            matched.add(raw)
                            break
        return s

    strong = [(raw, w) for key, (w, tparts, kind, raw) in terms.items() if kind in ("code", "trace") or (kind == "join")]
    strong_res = [(re.compile(r"\b%s\b" % re.escape(raw)), raw, w) for raw, w in strong if len(raw) >= 4][:60]
    title_res = []
    if on("titlerefs"):
        title_res = [(re.compile(r"\b%s\b" % re.escape(raw), re.I), raw, w) for key, (w, tparts, kind, raw) in terms.items()
                     if kind == "title" and len(raw) >= 4 and raw.lower() not in WEAK][:20]
    phrase_hits = {}
    for ph in phrases[:12]:
        ph = ph.strip()
        if len(ph) < 8:
            continue
        for p, t in file_text.items():
            if ph in t:
                for i, line in enumerate(file_lines[p]):
                    if ph in line:
                        phrase_hits.setdefault(p, []).append(i + 1)
    tb_files = set()
    for tp in tb_paths:
        tail = "/".join(tp.split("/")[-2:])
        for p in file_text:
            if ("/" + p).endswith("/" + tail):
                tb_files.add(p)

    frames = extract_frames(text) if on("tbf") else []
    frame_hit = {}
    for fi, (fp, fl, ff) in enumerate(frames):
        tail = "/".join(fp.split("/")[-2:])
        for p in file_text:
            if ("/" + p).endswith("/" + tail) or ("/" + p).endswith("/" + fp.split("/")[-1]) and p.count("/") == 0:
                frame_hit.setdefault((p, ff), []).append((fi, fl))
    inner_idx = max([fi for v in frame_hit.values() for fi, fl in v] or [-1])   # innermost frame inside the repo
    file_cache, fs_cache = R.file_cache, {}
    feats, matched_of, mult_of = {}, {}, {}
    for d in index:
        matched = set()
        s = name_score(d["name"], matched)
        if d["cls"]:
            s += 0.5 * name_score(d["cls"], matched)
        p = d["path"]
        if p not in file_cache:
            stem_ = os.path.basename(p)[:-3]
            if stem_ == "__init__":
                stem_ = p.split("/")[-2] if "/" in p else stem_
            file_cache[p] = stem_
        if p not in fs_cache:
            fs_cache[p] = 0.35 * name_score(file_cache[p], set())
        f = {"name": s}
        if s > 0:
            f["stem"] = fs_cache[p]
            s += fs_cache[p]
        body = "\n".join(file_lines[p][d["line"] - 1:d["own_end"]][:250])
        refs = 0.0
        for rx, raw, w in strong_res:
            if raw in body and rx.search(body):
                refs += 0.6 * w
                matched.add(raw)
        f["refs"] = min(refs, 6.0)
        s += f["refs"]
        if title_res and s > 0:
            tr = 0.0
            for rx, raw, w in title_res:
                if rx.search(body):
                    tr += 0.3 * w
                    matched.add(raw)
            f["tr"] = min(tr, 2.0)
            s += f["tr"]
        if p in phrase_hits and any(d["line"] <= ln <= d["own_end"] for ln in phrase_hits[p]):
            f["phr"] = 6.0
            matched.add("<text>")
        if p in tb_files:
            f["tb"] = 2.0
        if frame_hit:
            hits = frame_hit.get((p, d["name"]))
            if hits:
                inner = max(fi for fi, fl in hits) == inner_idx
                f["tbf"] = 1.0 if inner else 0.6
                if any(d["line"] <= fl <= d["end"] for fi, fl in hits):
                    f["tbf"] += 0.4
                if WEIGHTS.get("tbf"):
                    matched.add("<frame>")
        m = 1.0
        if d["kind"] == "var":
            m *= 0.6
        if d["name"].startswith("__") and d["name"].endswith("__") and d["name"] not in terms \
                and not (on("dunder") and d["own_end"] - d["line"] >= 20):   # long __init__/__rich_console__ hold logic
            m *= 0.5
        m *= file_factor(p)
        mult_of[d["id"]] = m
        feats[d["id"]] = f
        matched_of[d["id"]] = matched
    # v5 candidates: definitions with a positive base score (before coverage bonuses)
    base_total = {i: sum(WEIGHTS_V5[k] * v for k, v in f.items() if k in WEIGHTS_V5) * mult_of[i] for i, f in feats.items()}
    title_words = {raw for key, (w, tp, kind, raw) in terms.items() if kind == "title"}
    code_words = {raw for key, (w, tp, kind, raw) in terms.items() if kind in ("code", "trace")}
    file_terms = {}
    cand = {i for i, s in base_total.items() if s > 0}
    if on("multi"):   # files whose definitions match several distinct issue identifiers / title words
        for i in cand:
            if base_total[i] >= 1.0:
                file_terms.setdefault(index[i]["path"], set()).update(
                    m for m in matched_of[i] if m in code_words or m in title_words)
    out = {}
    for i in cand:
        f = feats[i]
        f["cov"] = float(len([m for m in matched_of[i] if m in title_words]))
        ft = file_terms.get(index[i]["path"])
        f["multi"] = min(3.0, 0.6 * (len(ft) - 1)) if ft else 0.0
    # ---- v6 signals
    extra = {}
    if any(use(k) for k in NEW_FEATS):
        if not R.built:
            build_chunks(R)
        bm, qw = bm25_scores(R, terms) if use("bm25") else ({}, {})
        mx = max(bm.values()) if bm else 1.0
        for i, v in bm.items():
            extra.setdefault(i, {})["bm25"] = v / mx
            matched_of.setdefault(i, set())
            mult_of.setdefault(i, 1.0)
        # stage-1 scores -> seeds
        s1 = {}
        for i in set(cand) | set(extra):
            f = feats.get(i, {})
            s1[i] = (sum(WEIGHTS_V5[k] * f.get(k, 0.0) for k in ("name", "stem", "refs", "tr", "phr", "tb")) * mult_of[i]
                     + 1.5 * f.get("cov", 0.0) + f.get("multi", 0.0) + STAGE1["bm25"] * extra.get(i, {}).get("bm25", 0.0))
        seeds, seen_q = [], set()
        for i, v in sorted(s1.items(), key=lambda x: -x[1]):
            if v <= 0:
                break
            key = (index[i]["path"], index[i].get("qual"))
            if key in seen_q:
                continue
            seen_q.add(key)
            seeds.append((i, v))
            if len(seeds) >= SEEDS:
                break
        if seeds:
            top = seeds[0][1]
            for rank, (i, v) in enumerate(seeds):
                w = v / top
                tag = "via " + index[i]["name"]
                cs = callee_ids(R, i)
                if use("gcall"):
                    for j, cw in cs:
                        extra.setdefault(j, {})
                        extra[j]["gcall"] = extra[j].get("gcall", 0.0) + w * cw
                        matched_of.setdefault(j, set())
                        if WEIGHTS.get("gcall"):
                            matched_of[j].add(tag)
                        mult_of.setdefault(j, 1.0)
                if use("g2"):
                    for j, cw in cs:
                        for k2, cw2 in callee_ids(R, j):
                            if k2 != i:
                                extra.setdefault(k2, {})
                                extra[k2]["g2"] = extra[k2].get("g2", 0.0) + w * cw * cw2 * 0.5
                                mult_of.setdefault(k2, 1.0)
                                matched_of.setdefault(k2, set())
                if use("gcaller"):
                    ensure_callers(R)
                    for j in R.callers.get(i, ())[:20]:
                        extra.setdefault(j, {})
                        extra[j]["gcaller"] = extra[j].get("gcaller", 0.0) + w / (1 + 0.25 * len(R.callers.get(i, ())))
                        mult_of.setdefault(j, 1.0)
                        matched_of.setdefault(j, set())
                if rank < 3 and use("sclass") and index[i]["cls"]:
                    for j, d in enumerate(index):
                        if d["path"] == index[i]["path"] and d["cls"] == index[i]["cls"] and j != i and d["kind"] != "var":
                            extra.setdefault(j, {})
                            extra[j]["sclass"] = max(extra[j].get("sclass", 0.0), w)
                            mult_of.setdefault(j, 1.0)
                            matched_of.setdefault(j, set())
                if rank == 0 and use("sfile"):
                    for j in range(len(index)):
                        if index[j]["path"] == index[i]["path"] and j != i and index[j]["kind"] != "var":
                            extra.setdefault(j, {})
                            extra[j]["sfile"] = w
                            mult_of.setdefault(j, 1.0)
                            matched_of.setdefault(j, set())
            if use("tcall"):
                ensure_tests(R)
            if use("tcall") and R.tests:
                sn = {index[i]["name"] for i, v in seeds[:3] if len(index[i]["name"]) >= 4}
                qstems = set(qw)
                tc = {}
                for p, tname, calls in R.tests:
                    rel = 1.0 if (calls & sn) else 0.0
                    ov = len([t for t in set(ident_tokens(tname)) if t in qstems])
                    rel += 0.5 * min(2, ov)
                    if rel <= 0:
                        continue
                    res = [j for n in calls for j in R.name_idx.get(n, ())[:3]]
                    for j in res:
                        tc[j] = tc.get(j, 0.0) + rel / (len(res) ** 0.5)
                if tc:
                    mxt = max(tc.values())
                    for j, v in tc.items():
                        extra.setdefault(j, {})["tcall"] = v / mxt
                        mult_of.setdefault(j, 1.0)
                        matched_of.setdefault(j, set())
    for i, e in extra.items():
        feats.setdefault(i, {}).update(e)
    allc = set(cand) | set(extra) | {i for i, f in feats.items() if f.get("tbf")}
    return {i: feats[i] for i in allc}, matched_of, mult_of, terms, phrase_hits


def score_feats(feats, mult_of, weights=None):
    W = weights or WEIGHTS
    out = {}
    for i, f in feats.items():
        txt = 0.0
        for k, v in f.items():
            if k in ("cov", "multi"):
                continue
            txt += W.get(k, 0.0) * v
        if W.get("lname") and f.get("name"):
            txt += W["lname"] * math.log1p(f["name"])
        s = txt * mult_of.get(i, 1.0) + W["cov"] * f.get("cov", 0.0) + W["multi"] * f.get("multi", 0.0)
        if s > 0:
            out[i] = s
    return out


def rank_defs(R, text):
    """-> (final, terms, phrase_hits): final = [(score, def, matched)] sorted best first (all scored definitions)"""
    feats, matched_of, mult_of, terms, phrase_hits = featurize(R, text)
    sc = score_feats(feats, mult_of)
    final = [(s, R.index[i], matched_of.get(i, set())) for i, s in sc.items()]
    final.sort(key=lambda x: (-x[0], x[1]["path"], x[1]["line"]))
    return final, terms, phrase_hits


def dedupe_top(final, k, R=None):
    """one entry per (path, class, name), best first.  With R: methods of the same class whose bodies are near copies of a better
    ranked sibling (FastAPI.get/post/put/..., Query/Form/File wrappers) are folded into it (d["similar"]) so they do not fill the list."""
    seen, top = set(), []
    if R is not None:
        R.similar, R.folded = {}, set()
    for s, d, matched in final:
        key = (d["path"], d["cls"], d["name"])
        if key in seen:
            continue
        seen.add(key)
        if R is not None and on("collapse") and on("v5") and getattr(R, "built", False) and d["kind"] == "def" and d["cls"]:
            fb = set(R.tf[d["id"]][3])
            rep = None
            if len(fb) >= 6:
                for s2, d2, m2 in top:
                    if d2["path"] == d["path"] and d2["cls"] == d["cls"] and d2["kind"] == "def":
                        fb2 = set(R.tf[d2["id"]][3])
                        if len(fb2) >= 6 and len(fb & fb2) >= 0.85 * len(fb | fb2):
                            rep = d2
                            break
            if rep is not None:
                R.similar.setdefault(rep["id"], []).append(d.get("qual") or d["name"])
                R.folded.add(d["id"])
                continue
        top.append((s, d, matched))
        if len(top) >= k:
            break
    return top


def pick_more(final, top, n=None, R=None):
    """the next best definitions listed compactly; the best definition of up to 4 not-yet-shown files comes first"""
    n = n or N_MORE
    shown = {(d["path"], d["cls"], d["name"]) for s, d, m in top}
    shown_files = {d["path"] for s, d, m in top}
    rest = dedupe_top(final, len(top) + 60)
    folded = getattr(R, "folded", set()) if R is not None else set()
    rest = [(s, d, m) for s, d, m in rest if (d["path"], d["cls"], d["name"]) not in shown and d["id"] not in folded]
    pick, fseen = [], set()
    for s, d, m in rest:                      # file diversity first
        if d["path"] not in shown_files and d["path"] not in fseen and len(fseen) < 4:
            fseen.add(d["path"])
            pick.append((s, d, m))
    for s, d, m in rest:
        if len(pick) >= n:
            break
        if not any(d is x[1] for x in pick):
            pick.append((s, d, m))
    pick = pick[:n]
    pick.sort(key=lambda x: -x[0])
    return pick


def more_candidates(final, top, n=None, R=None):
    pick = pick_more(final, top, n, R)
    by_file, order = {}, []
    for s, d, m in pick:
        if d["path"] not in by_file:
            by_file[d["path"]] = []
            order.append(d["path"])
        by_file[d["path"]].append("%d %s" % (d["line"], d.get("qual") or d["name"]))
    if not order:
        return []
    return ["More candidates: " + " | ".join("%s: %s" % (p, ", ".join(by_file[p])) for p in order)]


LF = {"rest": 0.3, "phrase": 0.3, "n": 3}


def likely_files(final, phrase_hits):
    """3 files to look at.  v5 summed the scores of the 60 best definitions per file (many small hits in utility scripts beat one
    strong hit) and added a flat +5 for text found, which is huge on the v6 score scale; v6: best definition of the file plus a
    fraction of its next two, text-found bonus relative to the best score."""
    if on("v5") and on("lf"):       # (SWE_LOCATE_OFF=v5 or lf: the old aggregation below)
        fscore = {}
        for s, d, m in final[:60]:
            fscore.setdefault(d["path"], []).append(s)
        top1 = final[0][0] if final else 0.0
        out = {}
        for p, ss in fscore.items():
            ss.sort(reverse=True)
            out[p] = ss[0] + LF["rest"] * sum(ss[1:1 + int(LF["n"]) - 1])
        for p, lns in phrase_hits.items():
            out[p] = out.get(p, 0.0) + LF["phrase"] * top1
        return [p for p, _ in sorted(out.items(), key=lambda x: (-x[1], x[0]))[:3]]
    fscore = {}
    for s, d, m in final[:60]:
        fscore[d["path"]] = fscore.get(d["path"], 0.0) + s
    for p, lns in phrase_hits.items():
        fscore[p] = fscore.get(p, 0.0) + 5
    return [p for p, _ in sorted(fscore.items(), key=lambda x: -x[1])[:3]]

# ----------------------------------------------------------------------------- v8: print the code of the best hit(s)
# After locate the agent's next move was `sed -n 'A,Bp'` on the best hit (one LLM turn).  v8 prints that code inline, line
# numbered, bounded.  SWE_LOCATE_OFF=body switches it off (output == v6); SWE_LOCATE_BODY="n=60,r2=1" overrides the knobs.
BODY = {"n": 60,          # max lines printed for the rank-1 definition
        "k": 2,           # how many of the best definitions get their code printed (1 = rank 1 only)
        "short": 25,      # a further definition is printed only when everything printed so far is at most this many lines
        "n_k": 30,        # max lines for each further definition
        "tot": 80,        # max lines printed over all definitions
        "head": 12,       # lines of a long definition always shown (signature + start)
        "ctx": 3,         # context lines around a best-matching line of a long definition
        "width": 150,     # longer lines are cut
        "total": 4800}    # max characters of the whole output (the harness cuts command output at 5000)
for _kv in os.environ.get("SWE_LOCATE_BODY", "").split(","):
    if "=" in _kv:
        _k, _v = _kv.split("=", 1)
        try:
            BODY[_k.strip()] = int(_v)
        except ValueError:
            pass


def body_on():
    return on("body") and on("v5")


def _span(d):
    """first and last line of the code to show for a definition (a class: header + class-level code up to its first method)"""
    a = d["line"]
    b = d.get("own_end", d["end"]) if d["kind"] == "class" else d["end"]
    return a, max(a, b)


def _line_scores(R, d, a, b, terms, phrase_hits):
    """per line of a..b: how many issue terms it mentions (+ a big bonus for lines that contain text quoted from the issue)"""
    lines = R.file_lines[d["path"]]
    want = []
    for key, (w, tparts, kind, raw) in terms.items():
        want.append((key, [stem(p) for p in tparts], w))
    hits = set(phrase_hits.get(d["path"], ()))
    sc = {}
    for ln in range(a, b + 1):
        text = lines[ln - 1]
        if not text.strip():
            continue
        toks = set()
        for ident in IDENT_RE.findall(text):
            toks.add(ident.lower())
            toks.update(ident_tokens(ident))
        s = 0.0
        for key, tps, w in want:
            if key in toks or (tps and all(t in toks for t in tps)):
                s += w
        if ln in hits:
            s += 5.0
        if s > 0:
            sc[ln] = s
    return sc


def _header_end(R, d, a, b):
    """last line of the signature (a def header may span several lines)"""
    lines = R.file_lines[d["path"]]
    depth = 0
    for ln in range(a, min(b, a + 14) + 1):
        t = lines[ln - 1].split("#")[0]
        depth += t.count("(") + t.count("[") + t.count("{") - t.count(")") - t.count("]") - t.count("}")
        if depth <= 0 and t.rstrip().endswith(":"):
            return ln
    return a


def _merge(segs):
    segs = sorted(segs)
    out = [list(segs[0])]
    for x, y in segs[1:]:
        if x <= out[-1][1] + 2:
            out[-1][1] = max(out[-1][1], y)
        else:
            out.append([x, y])
    return [(x, y) for x, y in out]


def _cost(segs):
    return sum(y - x + 1 for x, y in segs) + len(segs) - 1


def pick_segments(R, d, terms, phrase_hits, n):
    """-> [(start, end)] line ranges to print for definition d, at most n lines.  Short definitions are printed whole; long ones as
    the head (signature + first lines) plus windows around the lines that mention the most issue terms."""
    a, b = _span(d)
    if b - a + 1 <= n:
        return [(a, b)]
    head_end = min(b, a + max(1, int(BODY["head"])) - 1)
    head_end = max(head_end, min(_header_end(R, d, a, b), a + n // 2))
    segs = [(a, head_end)]
    sc = _line_scores(R, d, head_end + 1, b, terms, phrase_hits)
    for ln, s in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0])):
        if any(x <= ln <= y for x, y in segs):
            continue
        for c in (int(BODY["ctx"]), 1, 0):
            cand = _merge(segs + [(max(head_end + 1, ln - c), min(b, ln + c))])
            if _cost(cand) <= n:
                segs = cand
                break
        if _cost(segs) >= n - 1:
            break
    if len(segs) == 1:      # nothing matched: the head plus the lines that follow
        segs = [(a, min(b, a + n - 1))]
    return segs


def _is_stub(R, d):
    """an @overload stub or an empty placeholder: its body is only `...` / `pass`"""
    lines = R.file_lines[d["path"]]
    a, b = _span(d)
    he = _header_end(R, d, a, b)
    body = [l.strip() for l in lines[he:b] if l.strip()]
    if body and all(x in ("...", "pass") for x in body):
        return True
    prev = lines[a - 2].strip() if a >= 2 else ""
    return prev.startswith("@") and "overload" in prev


def expand_twins(R, d):
    """definitions that share (path, class, name) with d - typing overloads, `if version: def f / else: def f` - are folded into one
    ranking entry; print the real implementation(s), not the stub that happened to score best"""
    ids = R.name_idx.get(d["name"], ()) if hasattr(R, "name_idx") else ()
    tw = [R.index[i] for i in ids if R.index[i]["path"] == d["path"] and R.index[i]["cls"] == d["cls"] and R.index[i]["kind"] == d["kind"]]
    if len(tw) <= 1:
        return [d]
    real = [x for x in tw if not _is_stub(R, x)]
    if not real:
        return [d]
    if d in real:
        real = [d] + [x for x in real if x is not d]
    return sorted(real[:2], key=lambda x: x["line"])


def class_methods(final, d0, k):
    """a class hit: its header and attributes say little - print the best ranked method(s) of that class instead (the class itself when it has none)"""
    ms, seen = [], set()
    for s, d, m in final:
        if d["path"] == d0["path"] and d["cls"] == d0["name"] and d["kind"] == "def" and d["name"] not in seen:
            seen.add(d["name"])
            ms.append(d)
            if len(ms) >= k:
                break
    return ms or [d0]


def body_plan(R, top, terms, phrase_hits, n=None, final=None):
    """-> [(rank, d, [(start, end)])] the definitions whose code is printed (n: cap for rank 1, scaled down when the output is too long)"""
    plan, shown, spans = [], 0, []
    n1 = int(n if n is not None else BODY["n"])
    tot = min(int(BODY["tot"]), n1 + int(BODY["tot"]) - int(BODY["n"])) if n is not None else int(BODY["tot"])
    for rank, (s, d0, m) in enumerate(top[:max(1, int(BODY["k"]))], 1):
        base = class_methods(final, d0, 2 if rank == 1 else 1) if (final and on("cls") and d0["kind"] == "class") else [d0]
        for d in [x for b0 in base for x in (expand_twins(R, b0) if on("twins") else [b0])]:
            if not plan:
                n_this = min(n1, tot)
            else:
                if shown > BODY["short"]:
                    break
                n_this = min(int(BODY["n_k"]), tot - shown)
                if n_this < 8:
                    break
            segs = pick_segments(R, d, terms, phrase_hits, n_this)
            a2, b2 = segs[0][0], segs[-1][1]
            if any(d["path"] == pth and a2 <= b1 and a1 <= b2 for pth, a1, b1 in spans):
                continue      # overlaps code that is already shown (a method inside a printed class, ...)
            spans.append((d["path"], a2, b2))
            shown += sum(y - x + 1 for x, y in segs)
            plan.append((rank, d, segs))
    return plan


def render_body(R, plan):
    out = []
    for rank, d, segs in plan:
        lines = R.file_lines[d["path"]]
        a, b = _span(d)
        qual = d.get("qual") or d["name"]
        w = int(BODY["width"])
        out.append("# code of %s:%d-%d (rank %d, %s)" % (d["path"], a, b, rank, qual))
        prev = None
        for x, y in segs:
            if prev is not None and x > prev + 1:
                out.append("      ... lines %d-%d not shown" % (prev + 1, x - 1))
            for ln in range(x, y + 1):
                t = lines[ln - 1].rstrip()
                out.append("%5d: %s" % (ln, t if len(t) <= w else t[:w] + " ..."))
            prev = y
        if prev is not None and prev < b:
            out.append("      ... lines %d-%d not shown" % (prev + 1, b))
    return out


def render(R, top, final, terms, phrase_hits):
    root, files, file_lines = R.root, R.files, R.file_lines
    out = []
    if not top:
        out.append("No definition matches the issue words. Terms tried: " + ", ".join(sorted(t[3] for t in terms.values())[:25]))
    else:
        out.append("Top definitions for the issue (best first):")
        for i, (s, d, matched) in enumerate(top, 1):
            qual = d.get("qual") or ((d["cls"] + "." if d["cls"] else "") + d["name"])
            out.append("%d. %s:%d  %s %s   [%s]" % (i, d["path"], d["line"], d["kind"], qual,
                                                  ", ".join(sorted(matched))[:60]))
            lines = file_lines[d["path"]]
            shown = 0
            for ln in range(d["line"] - 1, min(len(lines), d["end"], d["line"] + 6)):
                if shown >= 2:
                    break
                if lines[ln].strip():
                    out.append("     %d: %s" % (ln + 1, lines[ln].rstrip()[:110]))
                    shown += 1
            if getattr(R, "similar", {}).get(d["id"]):
                out.append("     (near-identical siblings: %s)" % ", ".join(R.similar[d["id"]][:6]))
    if top and N_MORE and on("more"):
        out.extend(more_candidates(final, top, R=R))
    top_files = likely_files(final, phrase_hits)
    for p, lns in (list(phrase_hits.items()) if "v5" in OFF else sorted(phrase_hits.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:3]):
        out.append("text from the issue found in %s: line %s" % (p, ", ".join(str(x) for x in lns[:5])))
    if top_files:
        out.append("Likely files: " + ", ".join(top_files))
        out.append("Existing tests for them:")
        for p in top_files:
            ts = rank_tests(root, files, p, limit=2)
            out.append("  %s -> %s" % (p, ", ".join(t for _, t, _ in ts) if ts else "(none found)"))
    if top and body_on():
        return with_body(R, top, terms, phrase_hits, out, final)
    out.append("Next: read the best hit (sed -n 'A,Bp' file), state PLAN, edit it.")
    s = "\n".join(out)
    if len(s) > MAX_OUT:
        s = s[:MAX_OUT] + "\n...[truncated]"
    return s


LAST_PLAN = []
NEXT_BODY = ("Next: the code of the best hit is printed above with line numbers. If the buggy line is in it, state PLAN and edit "
             "it with those line numbers; otherwise sed -n 'A,Bp' file for another hit.")


def with_body(R, top, terms, phrase_hits, out, final=None):
    """the v6 output (cut at MAX_OUT) + the numbered code of the best hit(s) + the Next line; whole output <= BODY['total']"""
    main = "\n".join(out)
    if len(main) > MAX_OUT:
        main = main[:MAX_OUT] + "\n...[truncated]"
    n = int(BODY["n"])
    while True:
        plan = body_plan(R, top, terms, phrase_hits, n, final)
        s = main + "\n" + "\n".join(render_body(R, plan)) + "\n" + NEXT_BODY
        if len(s) <= BODY["total"] or n <= 10:
            break
        n -= 5
    if len(s) > BODY["total"]:
        s = s[:BODY["total"]] + "\n...[truncated]"
    global LAST_PLAN
    LAST_PLAN = plan      # (read by the benchmark)
    return s


def main(argv):
    opts, pos = parse_args(argv, BOOL_FLAGS)
    text = opts.get("issue") or opts.get("text") or opts.get("query") or opts.get("title") or " ".join(pos)
    k = int(opts.get("k") or 8)
    root = find_root(opts)
    if not text.strip():
        print("usage: args {\"issue\": \"<issue title and key lines>\"}")
        return
    R = load_repo(root, truthy(opts.get("tests", "")) or truthy(opts.get("all", "")))
    if not R.index:
        print("no python definitions found under %s" % root)
        return
    final, terms, phrase_hits = rank_defs(R, text)
    top = dedupe_top(final, k, R)
    print(render(R, top, final, terms, phrase_hits))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    try:
        main(sys.argv[1:])
    except Exception as e:   # never fail with a traceback the model has to parse
        print("locate.py internal error: %s: %s" % (type(e).__name__, str(e)[:300]))
