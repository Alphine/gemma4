"""AST-located, text-applied mutation operators for synthetic bug injection.

Sites are found on the AST, but every mutation is a small set of byte-span replacements on the original
source, so the rest of the file (formatting, comments, quotes) is untouched and the reverse diff (the gold
patch) is minimal. All offsets are UTF-8 byte offsets (what `ast` reports in col_offset).

    muts = find_mutations(source_text, "pkg/mod.py")
    new_text = apply_mutation(source_text, muts[0])

Operators (OPS): cmp_flip, boolop_flip, binop_flip, augassign_flip, const_off_by_one, bound_off_by_one,
index_flip, negate_cond, remove_stmt, remove_early_exit, swap_args, default_value, kw_bool_flip,
return_bool_flip, name_sibling, attr_sibling, drop_except, break_continue.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

OPS = ("cmp_flip", "boolop_flip", "binop_flip", "augassign_flip", "const_off_by_one", "bound_off_by_one",
       "index_flip", "negate_cond", "remove_stmt", "remove_early_exit", "swap_args", "default_value",
       "kw_bool_flip", "return_bool_flip", "name_sibling", "attr_sibling", "drop_except", "break_continue")

CMP_FLIP = {ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"), ast.Gt: (">", ">="), ast.GtE: (">=", ">"),
            ast.Eq: ("==", "!="), ast.NotEq: ("!=", "=="), ast.Is: ("is", "is not"),
            ast.IsNot: ("is not", "is"), ast.In: ("in", "not in"), ast.NotIn: ("not in", "in")}
CMP_TOKEN_RE = {"<": r"<(?!=)", "<=": r"<=", ">": r">(?!=)", ">=": r">=", "==": r"==", "!=": r"!=",
                "is": r"\bis\b(?!\s+not\b)", "is not": r"\bis\s+not\b", "in": r"(?<!not )\bin\b",
                "not in": r"\bnot\s+in\b"}
BINOP_FLIP = {ast.Add: ("+", "-"), ast.Sub: ("-", "+")}
BINOP_TOKEN_RE = {"+": r"\+(?!=)", "-": r"-(?!=)"}
AUG_FLIP = {ast.Add: ("+=", "-="), ast.Sub: ("-=", "+=")}
# sibling methods / builtins: a classic "called the wrong one" bug
NAME_SIBLINGS = {"startswith": "endswith", "endswith": "startswith", "lstrip": "rstrip", "rstrip": "lstrip",
                 "lower": "upper", "upper": "lower", "min": "max", "max": "min", "any": "all", "all": "any",
                 "find": "rfind", "rfind": "find", "index": "rindex", "rindex": "index", "ljust": "rjust",
                 "rjust": "ljust", "floor": "ceil", "ceil": "floor", "isupper": "islower", "islower": "isupper",
                 "keys": "values", "values": "keys", "popleft": "pop", "appendleft": "append",
                 "issubset": "issuperset", "issuperset": "issubset", "union": "intersection",
                 "intersection": "union", "isdigit": "isalpha", "isalpha": "isdigit", "getattr": "hasattr"}
SKIP_CALLS = {"print", "warn", "warning", "debug", "info", "error", "exception", "critical", "log",
              "isinstance", "issubclass", "setattr", "getattr", "hasattr", "super", "max", "min", "len",
              "format", "join", "append", "add", "update", "extend", "deprecated", "warnings"}
SKIP_STMT_CALLS = {"print", "warn", "warning", "debug", "info", "error", "exception", "critical", "log",
                   "deprecated", "_deprecated", "simplefilter", "filterwarnings"}


@dataclass
class Edit:
    start: int   # byte offset (inclusive)
    end: int     # byte offset (exclusive)
    new: bytes

    def to_json(self):
        return {"start": self.start, "end": self.end, "new": self.new.decode("utf-8")}

    @staticmethod
    def from_json(d):
        return Edit(d["start"], d["end"], d["new"].encode("utf-8"))


@dataclass
class Mutation:
    op: str
    file: str            # path relative to the repo root
    func: str            # qualified name of the innermost enclosing function
    lineno: int          # first mutated line (1-based)
    end_lineno: int      # last mutated line
    edits: list = field(default_factory=list)
    detail: str = ""     # internal description (never shown to the agent)
    has_doc: bool = False
    public: bool = True  # enclosing function/class names have no leading underscore
    fn_first: int = 0    # line span of the enclosing function (coverage fallback, held-out split)
    fn_last: int = 0

    @property
    def key(self):
        return f"{self.file}:{self.lineno}:{self.op}:{self.edits[0].start}:{self.edits[0].new!r}"

    def lines(self):
        return set(range(self.lineno, self.end_lineno + 1))

    def to_json(self):
        return {"op": self.op, "file": self.file, "func": self.func, "lineno": self.lineno,
                "end_lineno": self.end_lineno, "edits": [e.to_json() for e in self.edits], "detail": self.detail,
                "has_doc": self.has_doc, "public": self.public, "fn_first": self.fn_first, "fn_last": self.fn_last}

    @staticmethod
    def from_json(d):
        m = Mutation(d["op"], d["file"], d["func"], d["lineno"], d["end_lineno"],
                     [Edit.from_json(e) for e in d["edits"]], d.get("detail", ""), d.get("has_doc", False),
                     d.get("public", True), d.get("fn_first", 0), d.get("fn_last", 0))
        return m


def apply_edits(src: bytes, edits) -> bytes:
    out = src
    for e in sorted(edits, key=lambda e: e.start, reverse=True):
        out = out[:e.start] + e.new + out[e.end:]
    return out


def apply_mutation(text: str, mut: Mutation) -> str:
    return apply_edits(text.encode("utf-8"), mut.edits).decode("utf-8")


def overlaps(a: Mutation, b: Mutation) -> bool:
    if a.file != b.file:
        return False
    return not (a.end_lineno < b.lineno or b.end_lineno < a.lineno)


class _Src:
    def __init__(self, text: str):
        self.b = text.encode("utf-8")
        self.line_starts = [0]
        for i, ch in enumerate(self.b):
            if ch == 10:
                self.line_starts.append(i + 1)

    def off(self, lineno, col):
        return self.line_starts[lineno - 1] + col

    def start(self, node):
        return self.off(node.lineno, node.col_offset)

    def end(self, node):
        return self.off(node.end_lineno, node.end_col_offset)

    def seg(self, node):
        return self.b[self.start(node):self.end(node)]

    def line_bytes(self, lineno):
        s = self.line_starts[lineno - 1]
        e = self.line_starts[lineno] if lineno < len(self.line_starts) else len(self.b)
        return self.b[s:e]

    def indent_of(self, lineno):
        ln = self.line_bytes(lineno)
        return ln[:len(ln) - len(ln.lstrip(b" \t"))]

    def stmt_owns_lines(self, node):
        """True if the statement alone occupies its lines (nothing else before/after it on them)."""
        first = self.line_bytes(node.lineno)
        if first[:node.col_offset].strip():
            return False
        last = self.line_bytes(node.end_lineno)
        rest = last[node.end_col_offset:].strip()
        return not rest or rest.startswith(b"#")

    def lines_span(self, first, last):
        s = self.line_starts[first - 1]
        e = self.line_starts[last] if last < len(self.line_starts) else len(self.b)
        return s, e


def _blank_comments(region: bytes) -> bytes:
    return re.sub(rb"#[^\n]*", lambda m: b" " * len(m.group(0)), region)


def _single_token(src: _Src, a_end: int, b_start: int, pattern: str):
    """(start, end) of the only match of `pattern` in src[a_end:b_start] (comments ignored), else None."""
    region = _blank_comments(src.b[a_end:b_start])
    ms = list(re.finditer(pattern.encode(), region))
    if len(ms) != 1:
        return None
    return a_end + ms[0].start(), a_end + ms[0].end()


def _is_int_const(node):
    return isinstance(node, ast.Constant) and type(node.value) is int


def _neg_int(node):
    """-N literal"""
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and _is_int_const(node.operand))


def _is_str_like(node):
    return isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)))


def _docstring_node(fn):
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        return fn.body[0]
    return None


def _guard_test(test):
    """if TYPE_CHECKING / __debug__ / sys.version_info / __name__ == '__main__' / platform checks"""
    s = ast.unparse(test)
    return bool(re.search(r"TYPE_CHECKING|__debug__|version_info|__name__|sys\.platform|os\.name|PY2|PY3|PYPY",
                          s))


def _call_name(call):
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


class _Finder(ast.NodeVisitor):
    def __init__(self, src: _Src, relpath: str, tree):
        self.src, self.relpath, self.tree = src, relpath, tree
        self.out: list[Mutation] = []
        self.stack: list = []           # enclosing ClassDef/FunctionDef nodes
        self.class_attrs: dict = {}     # class node -> set of self.<attr> names assigned in it
        self.skip_nodes: set = set()

    # ---------------------------------------------------------------- helpers
    def fn(self):
        for n in reversed(self.stack):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return n
        return None

    def cls(self):
        for n in reversed(self.stack):
            if isinstance(n, ast.ClassDef):
                return n
        return None

    def qualname(self):
        return ".".join(n.name for n in self.stack)

    def add(self, op, node, edits, detail, first=None, last=None):
        fn = self.fn()
        if fn is None or not edits:
            return
        if fn.name.startswith("test"):
            return
        for e in edits:
            if self.src.b[e.start:e.end] == e.new:
                return
        first = first or node.lineno
        last = last or getattr(node, "end_lineno", first)
        self.out.append(Mutation(op=op, file=self.relpath, func=self.qualname(), lineno=first, end_lineno=last,
                                 edits=edits, detail=detail, has_doc=_docstring_node(fn) is not None,
                                 public=not any(n.name.startswith("_") and not n.name.startswith("__")
                                                for n in self.stack),
                                 fn_first=fn.lineno, fn_last=fn.end_lineno))

    # ---------------------------------------------------------------- traversal
    def visit_ClassDef(self, node):
        attrs = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self" \
                    and isinstance(sub.ctx, ast.Store):
                attrs.add(sub.attr)
        self.class_attrs[node] = attrs
        self.stack.append(node)
        for b in node.body:
            self.visit(b)
        self.stack.pop()

    def _visit_fn(self, node):
        # defaults belong to the enclosing scope textually but are the function's behaviour
        self.stack.append(node)
        self._defaults(node)
        doc = _docstring_node(node)
        for b in node.body:
            if b is doc:
                continue
            self.visit(b)
        self.stack.pop()

    visit_FunctionDef = _visit_fn
    visit_AsyncFunctionDef = _visit_fn

    def generic_visit(self, node):
        if self.fn() is None and not isinstance(node, (ast.Module, ast.ClassDef, ast.If, ast.Try)):
            # module/class level code outside functions: only descend into defs
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self.visit(child)
            return
        super().generic_visit(node)

    # ---------------------------------------------------------------- operators
    def _defaults(self, fn):
        a = fn.args
        pos = a.posonlyargs + a.args
        pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + \
            [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
        for arg, d in pairs:
            if isinstance(d, ast.Constant) and isinstance(d.value, bool):
                new = b"False" if d.value else b"True"
            elif _is_int_const(d) and self.src.seg(d) == str(d.value).encode():
                new = str(d.value + 1 if d.value <= 0 else d.value - 1).encode()
            else:
                continue
            self.add("default_value", d, [Edit(self.src.start(d), self.src.end(d), new)],
                     f"default of {arg.arg}", first=d.lineno, last=d.end_lineno)

    def visit_Compare(self, node):
        if len(node.ops) == 1 and type(node.ops[0]) in CMP_FLIP:
            old, new = CMP_FLIP[type(node.ops[0])]
            # comparisons to None/True/False flip is/==: keep only the semantic flips
            tok = _single_token(self.src, self.src.end(node.left), self.src.start(node.comparators[0]),
                                CMP_TOKEN_RE[old])
            if tok:
                self.add("cmp_flip", node, [Edit(tok[0], tok[1], new.encode())], f"{old} -> {new}")
        if len(node.ops) == 1 and type(node.ops[0]) in (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq):
            c = node.comparators[0]
            self._const_pm1(c)
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        old, new = ("and", "or") if isinstance(node.op, ast.And) else ("or", "and")
        if len(node.values) == 2:
            a, b = node.values
            tok = _single_token(self.src, self.src.end(a), self.src.start(b), rf"\b{old}\b")
            if tok:
                self.add("boolop_flip", node, [Edit(tok[0], tok[1], new.encode())], f"{old} -> {new}")
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if type(node.op) in BINOP_FLIP and not (_is_str_like(node.left) or _is_str_like(node.right)) \
                and not isinstance(node.left, (ast.List, ast.Tuple, ast.ListComp)) \
                and not isinstance(node.right, (ast.List, ast.Tuple, ast.ListComp)):
            old, new = BINOP_FLIP[type(node.op)]
            def _intish(n):
                return _is_int_const(n) or (isinstance(n, ast.Call) and _call_name(n) == "len")
            # `+` is often str/list concatenation: require evidence of integers; `-` is rarely non-numeric
            numeric = _intish(node.left) or _intish(node.right) or (
                isinstance(node.op, ast.Sub) and isinstance(node.right, (ast.Call, ast.Name, ast.Attribute,
                                                                         ast.Subscript)))
            tok = _single_token(self.src, self.src.end(node.left), self.src.start(node.right), BINOP_TOKEN_RE[old])
            if tok and numeric:
                self.add("binop_flip", node, [Edit(tok[0], tok[1], new.encode())], f"{old} -> {new}")
        if type(node.op) in (ast.Add, ast.Sub) and _is_int_const(node.right):
            self._const_pm1(node.right)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if type(node.op) in AUG_FLIP and not _is_str_like(node.value) and \
                not isinstance(node.value, (ast.List, ast.Tuple, ast.ListComp)):
            old, new = AUG_FLIP[type(node.op)]
            tok = _single_token(self.src, self.src.end(node.target), self.src.start(node.value), re.escape(old))
            if tok and (_is_int_const(node.value) or isinstance(node.value, (ast.Name, ast.Call, ast.Attribute,
                                                                             ast.Subscript, ast.BinOp))):
                self.add("augassign_flip", node, [Edit(tok[0], tok[1], new.encode())], f"{old} -> {new}")
        self._maybe_remove_stmt(node)
        self.generic_visit(node)

    def _const_pm1(self, c):
        if _is_int_const(c) and self.src.seg(c) == str(c.value).encode() and abs(c.value) < 10**6:
            new = c.value + 1 if c.value <= 0 else c.value - 1
            self.add("const_off_by_one", c, [Edit(self.src.start(c), self.src.end(c), str(new).encode())],
                     f"{c.value} -> {new}")

    def _bound_pm1(self, e, delta):
        seg = self.src.seg(e)
        if isinstance(e, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
            new = seg + (b" + 1" if delta > 0 else b" - 1")
        elif isinstance(e, ast.BinOp):
            new = b"(" + seg + b")" + (b" + 1" if delta > 0 else b" - 1")
        else:
            return
        self.add("bound_off_by_one", e, [Edit(self.src.start(e), self.src.end(e), new)],
                 f"bound {'+' if delta > 0 else '-'}1")

    def visit_Subscript(self, node):
        s = node.slice
        if isinstance(s, ast.Slice):
            for i, b in enumerate((s.lower, s.upper)):
                if b is None:
                    continue
                if _is_int_const(b):
                    self._const_pm1(b)
                else:
                    self._bound_pm1(b, +1 if i == 0 else -1)
        elif isinstance(node.ctx, ast.Load):
            if _is_int_const(s) and s.value == 0 and self.src.seg(s) == b"0":
                self.add("index_flip", s, [Edit(self.src.start(s), self.src.end(s), b"-1")], "[0] -> [-1]")
            elif _neg_int(s) and s.operand.value == 1 and self.src.seg(s).replace(b" ", b"") == b"-1":
                self.add("index_flip", s, [Edit(self.src.start(s), self.src.end(s), b"0")], "[-1] -> [0]")
            elif _is_int_const(s) and s.value in (1, 2) and self.src.seg(s) == str(s.value).encode():
                self._const_pm1(s)
        self.generic_visit(node)

    def visit_Call(self, node):
        name = _call_name(node)
        # range()/enumerate start bounds
        if name == "range" and isinstance(node.func, ast.Name) and node.args:
            arg = node.args[-1] if len(node.args) <= 2 else node.args[1]
            if _is_int_const(arg):
                self._const_pm1(arg)
            else:
                self._bound_pm1(arg, -1)
        # swapped positional args
        if name and name not in SKIP_CALLS and len(node.args) >= 2 and not node.keywords \
                and not any(isinstance(a, ast.Starred) for a in node.args[:2]):
            a, b = node.args[0], node.args[1]
            sa, sb = self.src.seg(a), self.src.seg(b)
            if sa != sb and not (isinstance(a, ast.Constant) and isinstance(b, ast.Constant)) \
                    and a.end_lineno == b.lineno and b"\n" not in sa + sb:
                self.add("swap_args", node, [Edit(self.src.start(a), self.src.end(a), sb),
                                             Edit(self.src.start(b), self.src.end(b), sa)],
                         f"swap args of {name}()", first=a.lineno, last=b.end_lineno)
        # keyword bool flip
        for kw in node.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                v = kw.value
                self.add("kw_bool_flip", v, [Edit(self.src.start(v), self.src.end(v),
                                                  b"False" if v.value else b"True")], f"{kw.arg}={v.value} flipped")
        # sibling builtin / method
        f = node.func
        if isinstance(f, ast.Name) and f.id in NAME_SIBLINGS and f.id in ("min", "max", "any", "all", "getattr"):
            if f.id != "getattr" or len(node.args) == 2:
                self.add("name_sibling", f, [Edit(self.src.start(f), self.src.end(f), NAME_SIBLINGS[f.id].encode())],
                         f"{f.id} -> {NAME_SIBLINGS[f.id]}")
        elif isinstance(f, ast.Attribute) and f.attr in NAME_SIBLINGS and f.attr not in ("getattr",):
            s = self.src.end(f) - len(f.attr.encode())
            if self.src.b[s:self.src.end(f)] == f.attr.encode():
                self.add("name_sibling", f, [Edit(s, self.src.end(f), NAME_SIBLINGS[f.attr].encode())],
                         f".{f.attr} -> .{NAME_SIBLINGS[f.attr]}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        c = self.cls()
        if c is not None and isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) \
                and node.value.id == "self" and node.attr in self.class_attrs.get(c, ()):
            sib = _pick_sibling(node.attr, self.class_attrs[c])
            if sib:
                s = self.src.end(node) - len(node.attr.encode())
                if self.src.b[s:self.src.end(node)] == node.attr.encode():
                    self.add("attr_sibling", node, [Edit(s, self.src.end(node), sib.encode())],
                             f"self.{node.attr} -> self.{sib}")
        self.generic_visit(node)

    def visit_If(self, node):
        if self.fn() is not None and not _guard_test(node.test):
            self._negate(node.test)
            # early exit: `if cond: return/raise/continue/break` without else
            if not node.orelse and len(node.body) == 1 and \
                    isinstance(node.body[0], (ast.Return, ast.Raise, ast.Continue, ast.Break)) and \
                    self.src.stmt_owns_lines(node):
                self._delete_stmt(node, "remove_early_exit", "drop early exit")
        self.generic_visit(node)

    def visit_While(self, node):
        if self.fn() is not None and not (isinstance(node.test, ast.Constant)):
            self._negate(node.test)
        self.generic_visit(node)

    def visit_IfExp(self, node):
        if self.fn() is not None:
            self._negate(node.test)
        self.generic_visit(node)

    def _negate(self, test):
        seg = self.src.seg(test)
        if b"\n" in seg:
            return
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            new = self.src.seg(test.operand)
            if isinstance(test.operand, (ast.BoolOp, ast.Compare, ast.IfExp, ast.NamedExpr)) and \
                    not new.startswith(b"("):
                new = b"(" + new + b")"
        elif isinstance(test, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):   # Compare: see cmp_flip
            new = b"not " + seg
        elif isinstance(test, ast.BoolOp):
            new = b"not (" + seg + b")"
        else:
            return
        self.add("negate_cond", test, [Edit(self.src.start(test), self.src.end(test), new)], "negated condition")

    def visit_Return(self, node):
        v = node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, bool):
            self.add("return_bool_flip", v, [Edit(self.src.start(v), self.src.end(v),
                                                  b"False" if v.value else b"True")], "return bool flipped")
        self.generic_visit(node)

    def visit_Break(self, node):
        self.add("break_continue", node, [Edit(self.src.start(node), self.src.end(node), b"continue")],
                 "break -> continue")

    def visit_Continue(self, node):
        self.add("break_continue", node, [Edit(self.src.start(node), self.src.end(node), b"break")],
                 "continue -> break")

    def visit_Try(self, node):
        if self.fn() is not None:
            for h in node.handlers:
                if not h.body or (len(h.body) == 1 and isinstance(h.body[0], ast.Raise) and h.body[0].exc is None):
                    continue
                first, last = h.body[0], h.body[-1]
                if not (self.src.stmt_owns_lines(first) and self.src.stmt_owns_lines(last)):
                    continue
                if h.body[0].lineno == h.lineno:   # `except X: stmt` on one line
                    continue
                s, e = self.src.lines_span(first.lineno, last.end_lineno)
                ind = self.src.indent_of(first.lineno)
                self.add("drop_except", h, [Edit(s, e, ind + b"raise\n")],
                         "handler body replaced by re-raise", first=first.lineno, last=last.end_lineno)
        self.generic_visit(node)

    visit_TryStar = visit_Try

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and _call_name(node.value) not in SKIP_STMT_CALLS:
            self._maybe_remove_stmt(node)
        self.generic_visit(node)

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], (ast.Attribute, ast.Subscript)):
            self._maybe_remove_stmt(node)
        self.generic_visit(node)

    def _maybe_remove_stmt(self, node):
        if self.fn() is None or not self.src.stmt_owns_lines(node):
            return
        self._delete_stmt(node, "remove_stmt", "statement removed")

    def _delete_stmt(self, node, op, detail):
        parent_body = self._parent_body(node)
        if parent_body is None:
            return
        s, e = self.src.lines_span(node.lineno, node.end_lineno)
        others = [b for b in parent_body if b is not node and not (
            isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str))]
        new = b"" if others else self.src.indent_of(node.lineno) + b"pass\n"
        self.add(op, node, [Edit(s, e, new)], detail)

    def _parent_body(self, node):
        for parent in ast.walk(self.fn() or self.tree):
            for fld in ("body", "orelse", "finalbody"):
                body = getattr(parent, fld, None)
                if isinstance(body, list) and any(b is node for b in body):
                    return body
            for h in getattr(parent, "handlers", []) or []:
                if any(b is node for b in h.body):
                    return h.body
            for case in getattr(parent, "cases", []) or []:
                if any(b is node for b in case.body):
                    return case.body
        return None


def _pick_sibling(attr, attrs):
    """A sibling attribute name: same leading underscores and a shared word, deterministic."""
    cands = sorted(a for a in attrs if a != attr and a.startswith("_") == attr.startswith("_"))
    words = set(re.split(r"_+", attr.strip("_"))) - {""}
    shared = [a for a in cands if words & (set(re.split(r"_+", a.strip("_"))) - {""})]
    if shared:
        return shared[0]
    # common pairs
    pairs = {"start": "end", "end": "start", "left": "right", "right": "left", "min": "max", "max": "min",
             "first": "last", "last": "first", "width": "height", "height": "width", "key": "value",
             "value": "key", "prefix": "suffix", "suffix": "prefix", "head": "tail", "tail": "head"}
    for a in cands:
        if pairs.get(attr.strip("_")) == a.strip("_"):
            return a
    return None


def find_mutations(text: str, relpath: str) -> list[Mutation]:
    """All mutation sites in `text` (inside function bodies only), deduplicated and in source order."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    src = _Src(text)
    f = _Finder(src, relpath, tree)
    f.visit(tree)
    seen, out = set(), []
    for m in f.out:
        k = (m.op, tuple((e.start, e.end, e.new) for e in m.edits))
        if k in seen:
            continue
        seen.add(k)
        out.append(m)
    out.sort(key=lambda m: (m.lineno, m.edits[0].start, m.op))
    return out


def mutation_compiles(text: str, mut: Mutation) -> bool:
    try:
        compile(apply_mutation(text, mut), mut.file, "exec")
        return True
    except (SyntaxError, ValueError):
        return False
