"""Pallas/TPU-specific mutation operators as AST-located text edits.

Pure python (no jax). A site = dict(op, variant, line, col, end_line, end_col, new, func, func_role, stmt_span, desc)
apply_site(src, site) returns the mutated source (line numbers preserved: edits are single-expression replacements).
"""
import ast

OPS = [
    "IDX_SWAP", "IDX_OFF", "IDX_DROP", "BLOCK_SHAPE", "GRID_OFF1", "GRID_CDIV", "GRID_PERM",
    "PROGID_AXIS", "WHEN_COND", "WHEN_DROP", "ACC_INIT", "REDUCE_AXIS", "KEEPDIMS", "DTYPE_ACC",
    "MASK_DROP", "CMP_FLIP", "DS_OFF", "LOOP_BOUND", "CONST_DROP", "SEM_PARALLEL",
]
# coarse bug class used in the report (pallas-specific vs generic numeric)
CLASS = {
    "IDX_SWAP": "indexing", "IDX_OFF": "indexing", "IDX_DROP": "indexing", "BLOCK_SHAPE": "blocking",
    "GRID_OFF1": "grid", "GRID_CDIV": "grid", "GRID_PERM": "grid", "PROGID_AXIS": "grid",
    "WHEN_COND": "accumulator", "WHEN_DROP": "accumulator", "ACC_INIT": "accumulator",
    "REDUCE_AXIS": "reduction", "KEEPDIMS": "reduction", "DTYPE_ACC": "precision",
    "MASK_DROP": "masking", "CMP_FLIP": "masking", "DS_OFF": "slicing", "LOOP_BOUND": "slicing",
    "CONST_DROP": "scaling", "SEM_PARALLEL": "hw_semantics",
}


def dotted(n):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        d = dotted(n.value)
        return (d + "." if d else "") + n.attr
    return ""


def _is(call, *suffixes):
    if not isinstance(call, ast.Call):
        return False
    d = dotted(call.func)
    return any(d == s or d.endswith("." + s) for s in suffixes)


def _span(src_lines, n):
    return (n.lineno, n.col_offset, n.end_lineno, n.end_col_offset)


def _text(lines, n):
    l0, c0, l1, c1 = n.lineno - 1, n.col_offset, n.end_lineno - 1, n.end_col_offset
    if l0 == l1:
        return lines[l0].encode()[c0:c1].decode()
    parts = [lines[l0].encode()[c0:].decode()] + lines[l0 + 1:l1] + [lines[l1].encode()[:c1].decode()]
    return "\n".join(parts)


def apply_site(src, s):
    lines = src.split("\n")
    l0, c0, l1, c1 = s["line"] - 1, s["col"], s["end_line"] - 1, s["end_col"]
    head = lines[l0].encode()[:c0].decode()
    tail = lines[l1].encode()[c1:].decode()
    new = head + s["new"] + tail
    return "\n".join(lines[:l0] + [new] + lines[l1 + 1:])


class _Ctx:
    def __init__(self, src):
        self.src = src
        self.lines = src.split("\n")
        self.tree = ast.parse(src)
        for p in ast.walk(self.tree):
            for c in ast.iter_child_nodes(p):
                c._parent = p
        self.kernel_names = set()
        self.max_axis = 0
        for n in ast.walk(self.tree):
            if _is(n, "pallas_call") and n.args:
                a = n.args[0]
                if isinstance(a, ast.Name):
                    self.kernel_names.add(a.id)
                elif _is(a, "partial") and a.args and isinstance(a.args[0], ast.Name):
                    self.kernel_names.add(a.args[0].id)
            if _is(n, "program_id") and n.args and isinstance(n.args[0], ast.Constant):
                self.max_axis = max(self.max_axis, n.args[0].value)
        # inner functions called from kernels are kernel-role too
        changed = True
        defs = {n.name: n for n in ast.walk(self.tree) if isinstance(n, ast.FunctionDef)}
        while changed:
            changed = False
            for kn in list(self.kernel_names):
                if kn in defs:
                    for c in ast.walk(defs[kn]):
                        if isinstance(c, ast.FunctionDef) and c.name not in self.kernel_names and c is not defs[kn]:
                            self.kernel_names.add(c.name)
                            changed = True
                        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in defs \
                                and c.func.id not in self.kernel_names:
                            self.kernel_names.add(c.func.id)
                            changed = True

    def func_of(self, n):
        p = n
        while p is not None:
            p = getattr(p, "_parent", None)
            if isinstance(p, ast.FunctionDef):
                return p
        return None

    def stmt_of(self, n):
        p = n
        while p is not None and not isinstance(p, ast.stmt):
            p = getattr(p, "_parent", None)
        return p

    def in_kernel(self, n):
        f = self.func_of(n)
        while f is not None:
            if f.name in self.kernel_names:
                return True
            f = self.func_of(f)
        return False

    def mk(self, op, variant, node, new, desc):
        f = self.func_of(node)
        st = self.stmt_of(node)
        l, c, el, ec = _span(self.lines, node)
        return dict(op=op, variant=variant, line=l, col=c, end_line=el, end_col=ec, new=new,
                    func=f.name if f else "<module>",
                    func_role="kernel" if self.in_kernel(node) else "wrapper",
                    stmt_span=[st.lineno, st.end_lineno] if st else [l, el],
                    orig=_text(self.lines, node), desc=desc)


def _blockspec_lambdas(call):
    # index_map lambda: second positional or kw index_map
    out = []
    for a in call.args[1:2]:
        if isinstance(a, ast.Lambda):
            out.append(a)
    for k in call.keywords:
        if k.arg == "index_map" and isinstance(k.value, ast.Lambda):
            out.append(k.value)
    return out


def find_sites(src):
    cx = _Ctx(src)
    S = []
    T = lambda n: _text(cx.lines, n)
    for n in ast.walk(cx.tree):
        # ---- BlockSpec: index_map and block_shape
        if _is(n, "BlockSpec"):
            for lam in _blockspec_lambdas(n):
                b = lam.body
                if isinstance(b, ast.Tuple) and len(b.elts) >= 2:
                    e = b.elts
                    texts = [T(x) for x in e]
                    for i in range(len(e)):
                        for j in range(i + 1, len(e)):
                            if texts[i] != texts[j]:
                                nt = list(texts)
                                nt[i], nt[j] = nt[j], nt[i]
                                S.append(cx.mk("IDX_SWAP", "swap%d%d" % (i, j), b, "(" + ", ".join(nt) + ")",
                                               "index_map returns elements %d and %d swapped" % (i, j)))
                                break
                        else:
                            continue
                        break
                if isinstance(b, ast.Tuple):
                    for i, x in enumerate(b.elts):
                        if isinstance(x, ast.Name):
                            S.append(cx.mk("IDX_OFF", "plus1_%d" % i, x, "%s + 1" % x.id, "index_map element %d off by one" % i))
                            S.append(cx.mk("IDX_DROP", "zero_%d" % i, x, "0", "index_map ignores grid axis %s" % x.id))
                        elif isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Name):
                            S.append(cx.mk("IDX_DROP", "zero_sub%d" % i, x.slice, "0", "scalar-prefetch index ignores grid axis"))
            if n.args and isinstance(n.args[0], ast.Tuple):
                for i, x in enumerate(n.args[0].elts):
                    if isinstance(x, ast.Constant) and x.value == 1:
                        continue
                    S.append(cx.mk("BLOCK_SHAPE", "half%d" % i, x, "(%s) // 2" % T(x), "block_shape dim %d halved" % i))
        # ---- grid
        if isinstance(n, ast.keyword) and n.arg == "grid" and isinstance(n.value, ast.Tuple):
            el = n.value.elts
            for i, x in enumerate(el):
                if _is(x, "cdiv") and len(x.args) == 2:
                    S.append(cx.mk("GRID_CDIV", "floor%d" % i, x, "(%s) // (%s)" % (T(x.args[0]), T(x.args[1])),
                                   "grid dim %d uses floor division instead of cdiv (ragged tail dropped)" % i))
                elif not (isinstance(x, ast.Constant) and x.value <= 1):
                    S.append(cx.mk("GRID_OFF1", "m1_%d" % i, x, "(%s) - 1" % T(x), "grid dim %d one too small" % i))
            tx = [T(x) for x in el]
            for i in range(len(el)):
                done = False
                for j in range(i + 1, len(el)):
                    if tx[i] != tx[j]:
                        nt = list(tx)
                        nt[i], nt[j] = nt[j], nt[i]
                        S.append(cx.mk("GRID_PERM", "perm%d%d" % (i, j), n.value, "(" + ", ".join(nt) + ("," if len(nt) == 1 else "") + ")",
                                       "grid dims %d and %d permuted without updating index_maps" % (i, j)))
                        done = True
                        break
                if done:
                    break
        # ---- program_id axis
        if _is(n, "program_id") and n.args and isinstance(n.args[0], ast.Constant) and cx.max_axis >= 1:
            k = n.args[0].value
            S.append(cx.mk("PROGID_AXIS", "axis%d_to_%d" % (k, (k + 1) % (cx.max_axis + 1)), n.args[0],
                           str((k + 1) % (cx.max_axis + 1)), "program_id axis %d -> %d" % (k, (k + 1) % (cx.max_axis + 1))))
        # ---- pl.when
        if _is(n, "when") and isinstance(getattr(n, "_parent", None), ast.FunctionDef) and n.args:
            c = n.args[0]
            S.append(cx.mk("WHEN_DROP", "unguard", n, "(lambda f: f())", "pl.when guard removed: body runs on every grid step"))
            if isinstance(c, ast.Compare) and len(c.comparators) == 1:
                comp = c.comparators[0]
                if isinstance(comp, ast.Constant) and isinstance(comp.value, int):
                    S.append(cx.mk("WHEN_COND", "const+1", comp, str(comp.value + 1), "pl.when condition compares against %d instead of %d" % (comp.value + 1, comp.value)))
                else:
                    S.append(cx.mk("WHEN_COND", "last-1", comp, "(%s) - 1" % T(comp), "pl.when condition off by one"))
        # ---- accumulator init
        if isinstance(n, ast.Call) and cx.in_kernel(n):
            d = dotted(n.func)
            if d.endswith("jnp.zeros_like"):
                S.append(cx.mk("ACC_INIT", "ones_like", n.func, "jnp.ones_like", "accumulator initialised with ones"))
            elif d.endswith("jnp.zeros"):
                S.append(cx.mk("ACC_INIT", "ones", n.func, "jnp.ones", "accumulator initialised with ones"))
            elif d.endswith("jnp.full") and len(n.args) >= 2 and isinstance(n.args[1], ast.UnaryOp):
                S.append(cx.mk("ACC_INIT", "neginf_to_0", n.args[1], "0.0", "running-max initialised with 0 instead of -inf"))
        # ---- reductions
        if isinstance(n, ast.keyword) and n.arg == "axis" and cx.in_kernel(n):
            v = n.value
            val = None
            if isinstance(v, ast.Constant) and isinstance(v.value, int):
                val = v.value
            elif isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub) and isinstance(v.operand, ast.Constant):
                val = -v.operand.value
            if val is not None:
                S.append(cx.mk("REDUCE_AXIS", "axis%d_to_%d" % (val, 0 if val != 0 else -1), v, "0" if val != 0 else "-1",
                               "reduction over axis %d instead of %d" % (0 if val != 0 else -1, val)))
        if isinstance(n, ast.keyword) and n.arg == "keepdims" and isinstance(n.value, ast.Constant) and n.value.value is True:
            S.append(cx.mk("KEEPDIMS", "false", n.value, "False", "keepdims dropped"))
        # ---- precision
        if isinstance(n, ast.keyword) and n.arg == "preferred_element_type" and cx.in_kernel(n):
            if dotted(n.value).endswith("float32"):
                S.append(cx.mk("DTYPE_ACC", "pet_bf16", n.value, "jnp.bfloat16", "matmul accumulates in bfloat16"))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "astype" and cx.in_kernel(n) \
                and n.args and dotted(n.args[0]).endswith("float32"):
            S.append(cx.mk("DTYPE_ACC", "astype_bf16", n.args[0], "jnp.bfloat16", "upcast to float32 replaced by bfloat16"))
        # ---- masks
        if _is(n, "where") and len(n.args) == 3 and cx.in_kernel(n):
            S.append(cx.mk("MASK_DROP", "where_keep_x", n, T(n.args[1]), "jnp.where mask removed (always takes the value branch)"))
        if isinstance(n, ast.Compare) and cx.in_kernel(n) and len(n.ops) == 1:
            p = getattr(n, "_parent", None)
            if isinstance(p, ast.Call) and _is(p, "when"):
                continue
            op = n.ops[0]
            rep = {ast.Lt: "<=", ast.LtE: "<", ast.GtE: ">", ast.Gt: ">="}.get(type(op))
            if rep:
                S.append(cx.mk("CMP_FLIP", rep, n, "%s %s %s" % (T(n.left), rep, T(n.comparators[0])), "comparison boundary flipped to '%s'" % rep))
        # ---- slicing / loop bounds
        if _is(n, "ds") and len(n.args) == 2:
            S.append(cx.mk("DS_OFF", "start+1", n.args[0], "(%s) + 1" % T(n.args[0]), "pl.ds start offset by one"))
        if _is(n, "fori_loop") and len(n.args) >= 2:
            S.append(cx.mk("LOOP_BOUND", "fori_hi-1", n.args[1], "(%s) - 1" % T(n.args[1]), "fori_loop upper bound one too small"))
        if isinstance(n, ast.Call) and dotted(n.func) == "range" and len(n.args) == 1 and cx.in_kernel(n):
            S.append(cx.mk("LOOP_BOUND", "range-1", n.args[0], "(%s) - 1" % T(n.args[0]), "unrolled loop one iteration short"))
        # ---- scale / eps
        if isinstance(n, ast.BinOp) and cx.in_kernel(n):
            def nm(x):
                return dotted(x).split(".")[-1].lower()
            for keep, drop in ((n.left, n.right), (n.right, n.left)):
                if isinstance(n.op, (ast.Mult, ast.Div)) and "scale" in nm(drop) and isinstance(drop, ast.Name):
                    S.append(cx.mk("CONST_DROP", "scale", n, T(keep), "softmax/attention scale factor dropped"))
                    break
                if isinstance(n.op, ast.Add) and isinstance(drop, ast.Name) and nm(drop) in ("eps", "epsilon"):
                    S.append(cx.mk("CONST_DROP", "eps", n, T(keep), "epsilon dropped"))
                    break
        # ---- hardware semantics control
        if isinstance(n, ast.Constant) and n.value == "arbitrary":
            S.append(cx.mk("SEM_PARALLEL", "arb->par", n, '"parallel"', "reduction grid axis marked parallel (dimension_semantics)"))
    # de-duplicate by (op, line, col, new)
    seen, out = set(), []
    for s in S:
        k = (s["op"], s["line"], s["col"], s["new"])
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def functions(src):
    """Top-level and nested function spans for function-level ground truth."""
    t = ast.parse(src)
    return [(n.name, n.lineno, n.end_lineno) for n in ast.walk(t) if isinstance(n, ast.FunctionDef)]


if __name__ == "__main__":
    import sys, glob, collections
    tot = collections.Counter()
    for p in sorted(glob.glob(sys.argv[1] + "/*.py")):
        s = open(p).read()
        ss = find_sites(s)
        for x in ss:
            compile(apply_site(s, x), p, "exec")  # syntax-validity of every mutant
        c = collections.Counter(x["op"] for x in ss)
        tot.update(c)
        print(p.split("/")[-1], len(ss), dict(c))
    print("TOTAL", sum(tot.values()), dict(tot))
