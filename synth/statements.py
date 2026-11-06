"""Deterministic problem-statement generator for synthetic tasks.

Everything is derived by code from the failing tests (their source, pytest's failure record) plus a fixed set
of generic templates defined in this file. No per-task text is written by hand or by an LLM.

Styles (chosen with a RNG seeded by the instance id):
  terse     title only, e.g. "`parse()` returns wrong result for negative numbers"
  observed  title + observed vs expected (from the assertion / exception message)
  repro     title + minimal repro snippet built from the failing test body + traceback tail

Never mentions the mutated file, its path or line numbers, the mutation operator, or the gold diff.
"""
from __future__ import annotations

import ast
import random
import re
import textwrap

from .patches import find_test_def, parse_nodeid

UNITTEST_DIFF_PREFIX = re.compile(r"^(Lists|Tuples|Sets|Dicts|Sequences|Items in the \w+ \w+) differ:\s*")


def postprocess_statement(text):
    """Cleanups applied after generation (also applied to already generated datasets, see README)."""
    return re.sub(r"`(Lists|Tuples|Sets|Dicts|Sequences) differ: ", "`", text)


STYLE_WEIGHTS = {"terse": 0.35, "observed": 0.30, "repro": 0.35}
BUILTIN_FIXTURES_OK = {"request"}   # fixtures we can drop from a repro if unused in the body
UNITTEST_ASSERTS = {
    "assertEqual": "{0} == {1}", "assertEquals": "{0} == {1}", "assertNotEqual": "{0} != {1}",
    "assertTrue": "{0}", "assertFalse": "not {0}", "assertIs": "{0} is {1}", "assertIsNot": "{0} is not {1}",
    "assertIsNone": "{0} is None", "assertIsNotNone": "{0} is not None", "assertIn": "{0} in {1}",
    "assertNotIn": "{0} not in {1}", "assertIsInstance": "isinstance({0}, {1})", "assertGreater": "{0} > {1}",
    "assertLess": "{0} < {1}", "assertGreaterEqual": "{0} >= {1}", "assertLessEqual": "{0} <= {1}",
    "assertListEqual": "{0} == {1}", "assertDictEqual": "{0} == {1}", "assertTupleEqual": "{0} == {1}",
    "assertSetEqual": "{0} == {1}", "assertSequenceEqual": "list({0}) == list({1})",
    "assertMultiLineEqual": "{0} == {1}", "assertAlmostEqual": "round({0} - {1}, 7) == 0",
}

T_WRONG = ["{sym} returns wrong result for {inp}", "{sym} gives incorrect output with {inp}",
           "Incorrect result from {sym} for {inp}", "{sym} produces unexpected output for {inp}",
           "{sym} mishandles {inp}"]
T_WRONG_NOINP = ["{sym} returns wrong result", "Wrong result from {sym}", "{sym} gives incorrect output",
                 "Unexpected output from {sym}", "{sym} behaves incorrectly"]
T_EXC = ["{sym} raises {exc} for {inp}", "{exc} from {sym} with {inp}", "{sym} fails with {exc} on {inp}"]
T_EXC_NOINP = ["{sym} raises {exc}", "Unexpected {exc} in {sym}", "{sym} crashes with {exc}"]
T_NORAISE = ["{sym} no longer raises {exc} for {inp}", "{sym} does not raise {exc} for invalid input",
             "Missing {exc} from {sym}"]
T_TOPIC = ["{topic} is broken", "{topic} does not work", "Regression: {topic}", "Bug: {topic}",
           "{topic} gives wrong result"]
T_TOPIC_SYM = ["{sym}: {topic} is broken", "{sym}: {topic} does not work", "Problem with {topic} in {sym}"]

B_OBS = ["When I call `{call}` I get `{obs}`, but I expected `{exp}`.",
         "**Observed:** `{obs}`\n**Expected:** `{exp}`\n\n(for `{call}`)",
         "`{call}` returns `{obs}`; it should return `{exp}`.",
         "Expected `{exp}` from `{call}`, got `{obs}` instead."]
B_OBS_NOCALL = ["I get `{obs}`, but I expected `{exp}`.", "**Observed:** `{obs}`\n**Expected:** `{exp}`",
                "Expected `{exp}`, got `{obs}`."]
B_EXC = ["Calling `{call}` raises an exception:\n\n```\n{exc}: {msg}\n```",
         "`{call}` fails with `{exc}: {msg}`.",
         "This call:\n\n```python\n{call}\n```\n\nraises\n\n```\n{exc}: {msg}\n```"]
B_EXC_NOCALL = ["It fails with:\n\n```\n{exc}: {msg}\n```", "I get `{exc}: {msg}`."]
B_NORAISE = ["`{call}` should raise `{exc}`, but it returns normally.",
             "I expected `{call}` to raise `{exc}`; no exception is raised."]
B_ASSERT = ["The result is not what I expect:\n\n```\n{msg}\n```", "This check fails:\n\n```\n{msg}\n```"]
C_INTRO = ["Minimal reproduction:", "Steps to reproduce:", "Repro:", "To reproduce:", "Example:"]
C_OUT = ["Output:", "Result:", "Actual output:", "I get:"]


# ----------------------------------------------------------------------------- helpers
def _clip(s, n=160):
    s = str(s)
    return s if len(s) <= n else s[:n - 3] + "..."


def _scrub(text, scrub_paths, roots=()):
    if not text:
        return text
    t = re.sub(r"0x[0-9a-fA-F]{6,}", "0x...", str(text))
    for r in sorted(roots, key=len, reverse=True):
        if r:
            t = t.replace(r.rstrip("/") + "/", "")
    return t


def _mentions(text, scrub_paths):
    for p in scrub_paths:
        base = p.rsplit("/", 1)[-1]
        if p in text or (base and re.search(rf"(?<![\w.]){re.escape(base)}", text)):
            return True
    return False


def humanize_test_name(name, classes=()):
    s = re.sub(r"^test_?", "", name)
    s = re.sub(r"_?\d+$", "", s)
    words = [w for w in re.split(r"_+", s) if w]
    if not words and classes:
        words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", re.sub(r"^Test", "", classes[-1]))
    return " ".join(w.lower() if not w.isupper() else w for w in words)


def _bound_names(tree, packages):
    """Names in the test module bound to the repo's package(s) (imports)."""
    names = {}
    pk = set(packages)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            root = node.module.split(".")[0]
            if root in pk:
                for a in node.names:
                    if a.name != "*":
                        names[a.asname or a.name] = f"{node.module}.{a.name}"
        elif isinstance(node, ast.ImportFrom) and node.level and node.module is None:
            pass
        elif isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root in pk:
                    names[a.asname or root] = a.asname and a.name or root
    return names


def _root_name(expr):
    while isinstance(expr, (ast.Attribute, ast.Call, ast.Subscript)):
        expr = expr.func if isinstance(expr, ast.Call) else expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def _stmt_containing(body, line):
    for st in body:
        if st.lineno <= line <= st.end_lineno:
            return st
    return None


def _innermost_stmt(node, line):
    best = node
    for sub in ast.walk(node):
        if isinstance(sub, ast.stmt) and sub.lineno <= line <= sub.end_lineno:
            if (sub.end_lineno - sub.lineno) <= (best.end_lineno - best.lineno):
                best = sub
    return best


_UNKNOWN = object()


def _classify_value(v):
    if v is _UNKNOWN:
        return None
    if isinstance(v, bool):
        return f"`{v}`"
    if v is None:
        return "`None`"
    if isinstance(v, str):
        if v == "":
            return "an empty string"
        if "\n" in v:
            return "multi-line input"
        if any(ord(c) > 127 for c in v):
            return "non-ASCII text"
        if v.strip() != v:
            return "a string with surrounding whitespace"
        if len(v) <= 24:
            return f"`{v!r}`"
        return "a long string"
    if isinstance(v, bytes):
        return "bytes input" if v else "empty bytes"
    if isinstance(v, int):
        if v < 0:
            return "negative numbers"
        if v == 0:
            return "zero"
        if abs(v) >= 10 ** 6:
            return "large numbers"
        return f"`{v}`"
    if isinstance(v, float):
        return "negative floats" if v < 0 else "float input"
    if isinstance(v, (list, tuple, set, frozenset)):
        kind = type(v).__name__
        return f"an empty {kind}" if not v else (f"a nested {kind}" if any(isinstance(x, (list, tuple, dict)) for x in v)
                                                 else f"a {kind} input")
    if isinstance(v, dict):
        return "an empty dict" if not v else ("nested dicts" if any(isinstance(x, (dict, list)) for x in v.values())
                                              else "a dict input")
    return None


# ----------------------------------------------------------------------------- analysis
class FailureView:
    """What we can derive from one failing test."""

    def __init__(self, failure, packages, scrub_paths, roots=()):
        self.f = failure
        self.packages = packages
        self.scrub_paths = scrub_paths
        self.roots = roots
        self.path, self.classes, self.name, self.param_id = parse_nodeid(failure["nodeid"])
        self.src = failure.get("test_src") or ""
        self.ok = False
        try:
            self.tree = ast.parse(self.src)
            self.fn, _ = find_test_def(self.tree, self.classes, self.name)
        except SyntaxError:
            self.tree, self.fn = None, None
        if self.fn is None:
            return
        self.ok = True
        self.body = [b for b in self.fn.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                                                     and isinstance(b.value.value, str))]
        self.bound = _bound_names(self.tree, packages)
        self.exc = failure.get("exc_type") or "AssertionError"
        self.msg = _scrub(failure.get("exc_msg") or "", scrub_paths, roots)
        self.fail_line = None
        for p, ln, fn_name in reversed(failure.get("tb") or []):
            if str(p).replace("\\", "/").endswith(self.path) and fn_name == self.name:
                self.fail_line = ln
                break
        if self.fail_line is None or not (self.fn.lineno <= self.fail_line <= self.fn.end_lineno):
            self.fail_line = self.body[-1].lineno if self.body else self.fn.lineno
        self.outer = _stmt_containing(self.body, self.fail_line)
        self.inner = _innermost_stmt(self.outer, self.fail_line) if self.outer is not None else None
        self.assigns = {}
        for st in self.body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                self.assigns[st.targets[0].id] = st.value
        self.mod_assigns = {n.targets[0].id: n.value for n in self.tree.body if isinstance(n, ast.Assign)
                            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
        self.params = failure.get("params") or {}
        self.call, self.sym = self._pick_call()

    def _resolve_value(self, node, depth=0):
        """literal value of an argument expression, else _UNKNOWN"""
        if depth > 3 or node is None:
            return _UNKNOWN
        try:
            return ast.literal_eval(node)
        except Exception:
            pass
        if isinstance(node, ast.Name):
            if node.id in self.params:
                try:
                    return ast.literal_eval(self.params[node.id])
                except Exception:
                    return _UNKNOWN
            if node.id in self.assigns:
                return self._resolve_value(self.assigns[node.id], depth + 1)
            if node.id in self.mod_assigns:
                return self._resolve_value(self.mod_assigns[node.id], depth + 1)
        return _UNKNOWN

    def _sym_of(self, call):
        f = call.func
        root = _root_name(f)
        if root is None:
            return None
        if root in self.bound:
            txt = ast.unparse(f)
            # pkg.mod.func(...) -> keep as written; Class(...).method -> Class.method
            txt = re.sub(r"\(.*?\)", "", txt)
            return txt
        # method on a local built from a package call: x = Foo(...); x.bar(...)
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            v = self.assigns.get(f.value.id)
            if isinstance(v, ast.Call):
                r2 = _root_name(v.func)
                if r2 in self.bound:
                    return re.sub(r"\(.*?\)", "", ast.unparse(v.func)) + "." + f.attr
        return None

    def _pick_call(self):
        for scope in (self.inner, self.outer):
            if scope is None:
                continue
            calls = [n for n in ast.walk(scope) if isinstance(n, ast.Call)]
            # prefer the outermost package call in the failing statement
            for c in sorted(calls, key=lambda c: (c.lineno, c.col_offset)):
                s = self._sym_of(c)
                if s:
                    return c, s
        for st in self.body:
            for c in [n for n in ast.walk(st) if isinstance(n, ast.Call)]:
                s = self._sym_of(c)
                if s:
                    return c, s
        return None, None

    def sym_md(self):
        if not self.sym:
            return None
        s = self.sym
        last = s.rsplit(".", 1)[-1]
        return f"`{s}`" if last[:1].isupper() and "." not in s else f"`{s}()`"

    def input_desc(self):
        c = self.call
        if c is None:
            return None
        for a in c.args:
            d = _classify_value(self._resolve_value(a))
            if d:
                return d
        for kw in c.keywords:
            v = self._resolve_value(kw.value)
            if isinstance(v, bool) and kw.arg:
                return f"`{kw.arg}={v}`"
            d = _classify_value(v)
            if d and kw.arg:
                return f"{d} as `{kw.arg}`"
        return None

    def call_text(self):
        if self.call is None:
            return None
        t = ast.unparse(self.call)
        return _clip(t, 140) if "\n" not in t else None

    def no_raise(self):
        return self.exc in ("Failed",) and "DID NOT RAISE" in (self.f.get("exc_msg") or "")

    def expected_exc(self):
        m = re.search(r"DID NOT RAISE <class '([\w.]+)'>", self.f.get("exc_msg") or "")
        return m.group(1).rsplit(".", 1)[-1] if m else None

    def observed_expected(self):
        """(observed, expected) parsed from the assertion explanation, else None."""
        if self.exc != "AssertionError":
            return None
        first = (self.msg.strip().splitlines() or [""])[0]
        m = re.match(r"^assert (.+?) == (.+)$", first)
        if m and len(first) < 400:
            obs, exp = m.group(1).strip(), m.group(2).strip()
            # left side should be the computed value: if it is the literal expected, swap
            return _clip(obs, 200), _clip(exp, 200)
        first = UNITTEST_DIFF_PREFIX.sub("", first)
        m = re.match(r"^(.+?) != (.+)$", first)        # unittest assertEqual(first, second)
        if m and len(first) < 400 and not first.startswith("assert "):
            return _clip(m.group(1).strip(), 200), _clip(m.group(2).strip(), 200)
        return None


# ----------------------------------------------------------------------------- repro snippet
def _convert_unittest(stmt_src):
    """self.assertX(a, b) -> assert ...; returns None if other `self` usage remains."""
    try:
        tree = ast.parse(stmt_src)
    except SyntaxError:
        return None

    class T(ast.NodeTransformer):
        def visit_Expr(self, node):
            c = node.value
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and \
                    isinstance(c.func.value, ast.Name) and c.func.value.id == "self" and c.func.attr in UNITTEST_ASSERTS:
                args = [ast.unparse(a) for a in c.args[:2]]
                fmt = UNITTEST_ASSERTS[c.func.attr]
                need = fmt.count("{1}") and 2 or 1
                if len(args) < need:
                    return node
                return ast.parse("assert " + fmt.format(*[f"({a})" if " " in a else a for a in args])).body[0]
            return node

    tree = T().visit(tree)
    out = ast.unparse(tree)
    if re.search(r"\bself\b", out):
        return None
    return out


def build_repro(view: FailureView):
    """(snippet, fail_line_in_snippet, failing_stmt_text) or None."""
    if not view.ok or view.outer is None:
        return None
    fn = view.fn
    argnames = [a.arg for a in fn.args.args + fn.args.kwonlyargs if a.arg != "self"]
    params = view.params
    for a in argnames:
        if a not in params and a not in BUILTIN_FIXTURES_OK:
            return None    # needs a fixture we cannot inline
    is_method = bool(view.classes)
    if is_method:
        # unittest-style setUp / pytest setup_method would be needed
        cls_body = view.tree.body
        for c in view.classes:
            cls = next((n for n in cls_body if isinstance(n, ast.ClassDef) and n.name == c), None)
            if cls is None:
                return None
            cls_body = cls.body
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in
               ("setUp", "setup_method", "setup", "setup_class", "setUpClass") for n in cls_body):
            return None
    if isinstance(fn, ast.AsyncFunctionDef):
        return None
    lines = view.src.splitlines()
    upto = [b for b in view.body if b.lineno <= view.outer.lineno]
    stmts = []
    for b in upto:
        seg = "\n".join(lines[b.lineno - 1:b.end_lineno])
        seg = textwrap.dedent(seg)
        if is_method or "self." in seg:
            conv = _convert_unittest(seg)
            if conv is None:
                return None
            seg = conv
        stmts.append(seg)
    if not stmts or sum(s.count("\n") + 1 for s in stmts) > 25:
        return None
    body_src = "\n".join(stmts)
    # names needed from module level
    try:
        used = {n.id for n in ast.walk(ast.parse(body_src)) if isinstance(n, ast.Name)}
    except SyntaxError:
        return None
    header, defs_needed = [], []
    mod_imports = [n for n in view.tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    bound_by_import = {}
    for imp in mod_imports:
        for a in imp.names:
            nm = (a.asname or a.name).split(".")[0]
            bound_by_import[nm] = imp
    mod_assigns = {}
    for n in view.tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            mod_assigns[n.targets[0].id] = n
    builtins_ = set(dir(__builtins__)) if isinstance(__builtins__, dict) is False else set(__builtins__)
    local = set()
    for st in ast.walk(ast.parse(body_src)):
        if isinstance(st, ast.Name) and isinstance(st.ctx, ast.Store):
            local.add(st.id)
        elif isinstance(st, (ast.FunctionDef, ast.ClassDef)):
            local.add(st.name)
        elif isinstance(st, ast.arg):
            local.add(st.arg)
        elif isinstance(st, ast.alias):
            local.add((st.asname or st.name).split(".")[0])
    param_lines = []
    for a in argnames:
        if a in params and a in used:
            try:
                ast.literal_eval(params[a])
            except Exception:
                return None
            param_lines.append(f"{a} = {params[a]}")
            local.add(a)
    imports_out = []
    for nm in sorted(used - local):
        if nm in bound_by_import:
            imp = bound_by_import[nm]
            if isinstance(imp, ast.ImportFrom) and imp.level:
                return None
            if isinstance(imp, ast.Import):
                imp_txt = "import " + ", ".join(
                    (a.name + (f" as {a.asname}" if a.asname else "")) for a in imp.names
                    if (a.asname or a.name).split(".")[0] == nm)
            else:
                imp_txt = f"from {imp.module} import " + ", ".join(
                    (a.name + (f" as {a.asname}" if a.asname else "")) for a in imp.names if (a.asname or a.name) == nm)
            if imp_txt not in imports_out:
                imports_out.append(imp_txt)
        elif nm in mod_assigns:
            a = mod_assigns[nm]
            if a.end_lineno - a.lineno > 4:
                return None
            seg = "\n".join(lines[a.lineno - 1:a.end_lineno])
            inner_used = {n.id for n in ast.walk(a.value) if isinstance(n, ast.Name)}
            for nm2 in sorted(inner_used):
                if nm2 in bound_by_import:
                    imp = bound_by_import[nm2]
                    if isinstance(imp, ast.ImportFrom) and not imp.level:
                        t = f"from {imp.module} import " + ", ".join(
                            (x.name + (f" as {x.asname}" if x.asname else "")) for x in imp.names
                            if (x.asname or x.name) == nm2)
                        if t not in imports_out:
                            imports_out.append(t)
                    elif isinstance(imp, ast.Import):
                        t = "import " + nm2
                        if t not in imports_out:
                            imports_out.append(t)
                elif nm2 not in builtins_:
                    return None
            defs_needed.append(seg)
        elif nm in builtins_ or nm in ("pytest",):
            if nm == "pytest" and "import pytest" not in imports_out:
                imports_out.append("import pytest")
        else:
            return None   # helper function/class from the test module or a conftest fixture
    header = imports_out + ([""] if imports_out else []) + defs_needed + param_lines
    snippet = "\n".join(header + ([""] if header and header[-1] != "" else []) + stmts).strip() + "\n"
    if len(snippet) > 1800:
        return None
    try:
        compile(snippet, "repro.py", "exec")
    except SyntaxError:
        return None
    last_stmt = stmts[-1]
    fail_line = snippet.rstrip("\n").count("\n") + 1 - last_stmt.count("\n")
    # point at the innermost failing line when the last statement spans several lines
    if view.inner is not None and view.inner is not view.outer and view.outer.lineno < view.fail_line:
        fail_line += min(view.fail_line - view.outer.lineno, last_stmt.count("\n"))
    shown = snippet.splitlines()[fail_line - 1].strip() if fail_line - 1 < len(snippet.splitlines()) else last_stmt
    return snippet, fail_line, shown


def traceback_tail(view: FailureView, fail_line, shown, rel_root_tb):
    """Python-style traceback: repro frame + up to 2 library frames not in the mutated files + exception."""
    out = ["Traceback (most recent call last):", f'  File "repro.py", line {fail_line}, in <module>', f"    {shown}"]
    lib = []
    for p, ln, name in rel_root_tb:
        lib.append((p, ln, name))
    lib = [x for x in lib if not _mentions(x[0], view.scrub_paths)][-2:]
    for p, ln, name in lib:
        out.append(f'  File "{p}", line {ln}, in {name}')
    msg = view.msg.strip()
    if view.exc == "AssertionError":
        out.append("AssertionError")
    else:
        first = msg.splitlines()[0] if msg else ""
        out.append(f"{view.exc}: {_clip(first, 300)}" if first else view.exc)
    return "\n".join(out)


# ----------------------------------------------------------------------------- generator
def mutated_modules(scrub_paths):
    """'src/pkg/sub/mod.py' -> ['pkg.sub.mod'] (package __init__ -> the package; top-level packages kept)."""
    out = []
    for p in scrub_paths:
        q = re.sub(r"^(src|lib)/", "", p).removesuffix(".py").removesuffix("/__init__")
        mod = q.replace("/", ".")
        if "." in mod:          # never strip a bare top-level package name like `parse`
            out.append(mod)
    return sorted(out, key=len, reverse=True)


def _title(view, rng):
    sym = view.sym_md()
    inp = view.input_desc()
    topic = humanize_test_name(view.name, view.classes)
    exc = view.exc
    cands = []
    if view.no_raise() and sym:
        e = view.expected_exc() or "an error"
        cands += [t.format(sym=sym, exc=f"`{e}`", inp=inp or "invalid input") for t in T_NORAISE]
    elif exc not in ("AssertionError", "Failed") and sym:
        cands += [t.format(sym=sym, exc=f"`{exc}`", inp=inp) for t in (T_EXC if inp else T_EXC_NOINP)]
    elif sym:
        cands += [t.format(sym=sym, inp=inp) for t in (T_WRONG if inp else T_WRONG_NOINP)]
    if topic and len(topic.split()) >= 2:
        if sym:
            cands += [t.format(sym=sym, topic=topic) for t in T_TOPIC_SYM]
        else:
            cands += [t.format(topic=topic) for t in T_TOPIC]
    if not cands:
        cands = [f"{topic or view.name} fails"]
    t = rng.choice(cands)
    return t[0].upper() + t[1:]


def _observed_body(view, rng):
    call = view.call_text()
    oe = view.observed_expected()
    if view.no_raise():
        e = view.expected_exc()
        if e and call:
            return rng.choice(B_NORAISE).format(call=call, exc=e)
        return None
    if oe and oe[0] != oe[1]:
        obs, exp = oe
        if _mentions(obs + exp, view.scrub_paths):
            return None
        return (rng.choice(B_OBS) if call else rng.choice(B_OBS_NOCALL)).format(call=call, obs=obs, exp=exp)
    if view.exc not in ("AssertionError", "Failed"):
        msg = (view.msg.strip().splitlines() or [""])[0]
        if _mentions(msg, view.scrub_paths):
            msg = ""
        msg = _clip(msg, 300)
        if call:
            return rng.choice(B_EXC).format(call=call, exc=view.exc, msg=msg).replace(": \n", "\n").replace(": `", "`")
        return rng.choice(B_EXC_NOCALL).format(exc=view.exc, msg=msg)
    msg = "\n".join(view.msg.strip().splitlines()[:8])
    if msg and not _mentions(msg, view.scrub_paths) and len(msg) < 900:
        return rng.choice(B_ASSERT).format(msg=msg)
    return None


def _repro_body(view, rng, rel_tb):
    r = build_repro(view)
    if r is None:
        return None
    snippet, fail_line, shown = r
    tb = traceback_tail(view, fail_line, shown, rel_tb)
    extra = ""
    oe = view.observed_expected()
    if view.exc == "AssertionError" and oe and oe[0] != oe[1] and not _mentions(oe[0] + oe[1], view.scrub_paths):
        extra = rng.choice(["\n\nThe left-hand side is `{obs}` instead of `{exp}`.",
                            "\n\nI get `{obs}` where `{exp}` is expected."]).format(obs=oe[0], exp=oe[1])
    return (f"{rng.choice(C_INTRO)}\n\n```python\n{snippet}```\n\n{rng.choice(C_OUT)}\n\n```\n{tb}\n```{extra}")


def _module_bases(scrub_paths):
    """bare module names of the mutated files ('pkg/class_registry.py' -> 'class_registry'; not __init__)."""
    out = set()
    for p in scrub_paths:
        b = p.rsplit("/", 1)[-1].removesuffix(".py")
        if b != "__init__":
            out.add(b)
    return out


def strip_module(sym, scrub_paths):
    """`pkg.mutated_mod.func` / `mutated_mod.func` -> `func`: never name the module that holds the bug."""
    if not sym:
        return sym
    for mod in mutated_modules(scrub_paths):
        if sym.startswith(mod + "."):
            sym = sym[len(mod) + 1:]
    parts = sym.split(".")
    bases = _module_bases(scrub_paths)
    for k in range(len(parts) - 1, -1, -1):
        if k < len(parts) - 1 and parts[k] in bases:
            return ".".join(parts[k + 1:])
    return sym


def _leaks(text, scrub_paths):
    if _mentions(text, scrub_paths):
        return True
    return any(re.search(rf"(?<![\w]){re.escape(b)}\.\w", text) for b in _module_bases(scrub_paths))


def _pick_view(failures, packages, scrub_paths, roots, rng):
    views = [FailureView(f, packages, scrub_paths, roots) for f in sorted(failures, key=lambda f: f["nodeid"])]
    good = [v for v in views if v.ok]
    if not good:
        raise ValueError("no analysable failing test")
    # prefer a test whose failure names a package symbol, then the shortest test body
    good.sort(key=lambda v: (v.sym is None, (v.fn.end_lineno - v.fn.lineno), v.f["nodeid"]))
    view = good[0] if rng.random() < 0.7 else rng.choice(good)
    view.sym = strip_module(view.sym, scrub_paths)
    return view


def _rel_tb(view, scrub_paths, roots):
    out = []
    for p, ln, name in view.f.get("tb") or []:
        rp = _scrub(str(p).replace("\\", "/"), scrub_paths, roots)
        parts = rp.split("/")
        if (rp.startswith("/") or ":" in parts[0] or "site-packages" in rp or rp.startswith("<")
                or rp.endswith(view.path) or any(x in ("tests", "test", "testing") for x in parts[:-1])
                or parts[-1].startswith("test_") or parts[-1] == "conftest.py"):
            continue
        out.append((rp, ln, name))
    return out


def _body(view, style, rng, scrub_paths, roots):
    """(body or None, effective style)"""
    if style == "terse":
        return None, "terse"
    body = None
    if style == "repro":
        body = _repro_body(view, rng, _rel_tb(view, scrub_paths, roots))
        if body is None:
            style = "observed"
    if style == "observed":
        body = _observed_body(view, rng)
        if body is None:
            body = _repro_body(view, rng, _rel_tb(view, scrub_paths, roots))
            style = "repro" if body else "terse"
    if body is not None and _mentions(body, scrub_paths):   # quoted test code may use module.attr
        return None, "terse"
    return body, style


def generate(instance_id, failures, packages, scrub_paths, roots=(), force_style=None, groups=None):
    """-> (problem_statement, style, info). `failures`: one dict per failing test (see FailureView).
    `groups`: for multi-hunk tasks, lists of nodeids failing because of each hunk; every group gets described
    (title, plus its body for non-terse styles)."""
    rng = random.Random(f"stmt:{instance_id}")
    style = force_style
    if style is None:
        if rng.random() < STYLE_WEIGHTS["terse"]:
            style = "terse"
        else:
            style = "repro" if rng.random() < STYLE_WEIGHTS["repro"] / (1 - STYLE_WEIGHTS["terse"]) else "observed"
    by_id = {f["nodeid"]: f for f in failures}
    groups = [list(g) for g in (groups or [list(by_id)]) if g]
    parts, eff, syms = [], [], []
    used_tests = set()
    for gi, g in enumerate(groups):
        fl = [by_id[n] for n in g if n in by_id and n not in used_tests] or [by_id[n] for n in g if n in by_id]
        view = _pick_view(fl, packages, scrub_paths, roots, rng)
        used_tests |= set(g)
        title = _title(view, rng)
        if _leaks(title, scrub_paths):
            view.sym = None
            title = _title(view, rng)
            if _leaks(title, scrub_paths):
                raise ValueError("title mentions the mutated module")
        body, st = _body(view, style, rng, scrub_paths, roots)
        parts.append((title, body))
        eff.append(st)
        syms.append(view.sym)
    if style == "terse" or all(b is None for _, b in parts):
        text = parts[0][0] + "".join("; also " + t[0].lower() + t[1:] for t, _ in parts[1:])
        style_out = "terse"
    else:
        chunks = [parts[0][0] + ("\n\n" + parts[0][1] if parts[0][1] else "")]
        for t, b in parts[1:]:
            chunks.append(rng.choice(["A second problem: ", "Also: ", "Related: "]) + t[0].lower() + t[1:]
                          + ("\n\n" + b if b else ""))
        text = "\n\n".join(chunks)
        style_out = next((e for e in eff if e != "terse"), "terse")
    if _mentions(text, scrub_paths):
        raise ValueError("statement mentions the mutated file")
    return postprocess_statement(text.strip()) + "\n", style_out, {"symbol": syms[0], "symbols": syms, "n_failures": len(failures),
                                            "n_groups": len(groups)}
