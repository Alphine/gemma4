"""AST call-graph builder for the Candidate A study (see PREREG.md section 4).

build_graph(files, is_test) -> G with
  G.nodes   list of dict(id, path, qual, name, line, end, cls (qualname of the enclosing class or None), is_test, kind)
  G.adj     list[set[int]] callee ids per node (caller -> callee, directed)
  G.radj    list[set[int]] reversed
  G.stats   dict(n_calls, n_direct, n_fallback, n_unresolved, n_files, n_parse_fail)
Resolution order (PREREG 4): scope/module/import/class, self./cls. via class MRO, module alias attr, Class.method, then name
fallback to at most FALLBACK_MAX non-test defs with the same bare name.
"""
import ast
from collections import deque

FALLBACK_MAX = 3


def module_of(path):
    p = path[:-3] if path.endswith(".py") else path
    parts = p.split("/")
    if parts and parts[0] in ("src", "lib"):
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class _G(object):
    pass


def _dotted(node):
    """a.b.c -> 'a.b.c' for Name/Attribute chains, else None"""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class _Collector(ast.NodeVisitor):
    """collect the calls of ONE function body, not descending into nested defs/classes"""

    def __init__(self):
        self.calls = []

    def visit_FunctionDef(self, node):
        for d in node.decorator_list:
            self.visit(d)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        for d in node.decorator_list:
            self.visit(d)

    def visit_Call(self, node):
        self.calls.append(node)
        self.generic_visit(node)


def build_graph(files, is_test):
    """files: {relpath: source text}. is_test(relpath) -> bool"""
    G = _G()
    nodes, classes = [], {}                       # classes: (path, qual) -> dict(bases=[dotted], methods={name: node id})
    by_path_qual = {}                             # (path, qual) -> node id
    by_name = {}                                  # bare name -> [node id] (non-test defs)
    top_defs = {}                                 # path -> {name: ('def', id) | ('class', (path, qual))}
    trees, imports, stats = {}, {}, {"n_calls": 0, "n_direct": 0, "n_fallback": 0, "n_unresolved": 0,
                                     "n_files": len(files), "n_parse_fail": 0}
    mod2path = {}
    for p in files:
        mod2path.setdefault(module_of(p), p)

    def walk(path, body, stack, in_func, cls_stack, tflag):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = ".".join([n for n, _ in stack] + [node.name])
                cls = cls_stack[-1] if cls_stack and stack and stack[-1][1] == "class" else None
                nid = len(nodes)
                nodes.append({"id": nid, "path": path, "qual": qual, "name": node.name, "line": node.lineno,
                              "end": getattr(node, "end_lineno", node.lineno), "cls": cls, "is_test": tflag,
                              "kind": "def", "ast": node, "stack": [n for n, _ in stack],
                              "cls_self": cls_stack[-1] if cls_stack else None})
                by_path_qual[(path, qual)] = nid
                if not tflag:
                    by_name.setdefault(node.name, []).append(nid)
                if cls is not None:
                    classes[(path, cls)]["methods"].setdefault(node.name, nid)
                elif not stack:
                    top_defs[path][node.name] = ("def", nid)
                walk(path, node.body, stack + [(node.name, "def")], True, cls_stack, tflag)
            elif isinstance(node, ast.ClassDef):
                qual = ".".join([n for n, _ in stack] + [node.name])
                classes[(path, qual)] = {"bases": [_dotted(b) for b in node.bases if _dotted(b)], "methods": {},
                                         "name": node.name, "path": path}
                if not stack:
                    top_defs[path][node.name] = ("class", (path, qual))
                walk(path, node.body, stack + [(node.name, "class")], False, cls_stack + [qual], tflag)
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for sub in ("body", "orelse", "finalbody"):
                    walk(path, getattr(node, sub, []) or [], stack, in_func, cls_stack, tflag)
                for h in getattr(node, "handlers", []) or []:
                    walk(path, h.body, stack, in_func, cls_stack, tflag)

    for path, text in files.items():
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            stats["n_parse_fail"] += 1
            continue
        trees[path] = tree
        top_defs[path] = {}
        walk(path, tree.body, [], False, [], is_test(path))

    # imports per module (top level and anywhere: we take every import statement of the file)
    for path, tree in trees.items():
        tab = {}
        pkg = module_of(path).split(".") if path.endswith("__init__.py") else module_of(path).split(".")[:-1]
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    full = a.name
                    if a.asname:
                        tab[a.asname] = ("module", full)
                    else:
                        tab[full.split(".")[0]] = ("module", full.split(".")[0])
                        tab[full] = ("module", full)
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    base = pkg[:len(pkg) - (n.level - 1)] if n.level > 1 else pkg
                    mod = ".".join(base + ([n.module] if n.module else []))
                else:
                    mod = n.module or ""
                for a in n.names:
                    if a.name == "*":
                        continue
                    nm = a.asname or a.name
                    sub = (mod + "." + a.name) if mod else a.name
                    if sub in mod2path:
                        tab[nm] = ("module", sub)
                    else:
                        tab[nm] = ("symbol", mod, a.name)
        imports[path] = tab

    def class_lookup(path, dotted):
        """resolve a class reference (dotted name used in `path`) to (path, qual) or None"""
        head, _, rest = dotted.partition(".")
        if not rest:
            ent = top_defs.get(path, {}).get(head)
            if ent and ent[0] == "class":
                return ent[1]
            imp = imports.get(path, {}).get(head)
            if imp and imp[0] == "symbol" and imp[1] in mod2path:
                ent = top_defs.get(mod2path[imp[1]], {}).get(imp[2])
                if ent and ent[0] == "class":
                    return ent[1]
                # re-exported through a package __init__: follow one more import level
                p2 = mod2path[imp[1]]
                imp2 = imports.get(p2, {}).get(imp[2])
                if imp2 and imp2[0] == "symbol" and imp2[1] in mod2path:
                    ent = top_defs.get(mod2path[imp2[1]], {}).get(imp2[2])
                    if ent and ent[0] == "class":
                        return ent[1]
            return None
        imp = imports.get(path, {}).get(head)
        if imp and imp[0] == "module":
            m = imp[1] + "." + ".".join(rest.split(".")[:-1]) if rest.count(".") else imp[1]
            m = m.rstrip(".")
            if m in mod2path:
                ent = top_defs.get(mod2path[m], {}).get(rest.split(".")[-1])
                if ent and ent[0] == "class":
                    return ent[1]
        return None

    def mro_method(cls_key, name, depth=0, seen=None):
        seen = seen or set()
        if cls_key is None or cls_key in seen or depth > 6:
            return None
        seen.add(cls_key)
        c = classes.get(cls_key)
        if c is None:
            return None
        if name in c["methods"]:
            return c["methods"][name]
        for b in c["bases"]:
            bk = class_lookup(cls_key[0], b)
            r = mro_method(bk, name, depth + 1, seen)
            if r is not None:
                return r
        return None

    def init_of(ck):
        m = mro_method(ck, "__init__")
        return m if m is not None else mro_method(ck, "__call__")

    def module_def(modname, name):
        p = mod2path.get(modname)
        if p is None:
            return []
        ent = top_defs.get(p, {}).get(name)
        if ent is None:
            imp = imports.get(p, {}).get(name)          # re-export
            if imp and imp[0] == "symbol" and imp[1] in mod2path and imp[1] != modname:
                ent = top_defs.get(mod2path[imp[1]], {}).get(imp[2])
            if ent is None:
                return []
        if ent[0] == "def":
            return [ent[1]]
        m = init_of(ent[1])
        return [m] if m is not None else []

    def resolve(call, caller):
        path, stack = caller["path"], caller["stack"]
        f = call.func
        tab = imports.get(path, {})
        if isinstance(f, ast.Name):
            n = f.id
            # enclosing function scopes (innermost first), then module level
            for k in range(len(stack), -1, -1):
                pref = stack[:k]
                if k > 0 and (path, ".".join(pref)) in classes:
                    continue                                        # class bodies are not enclosing scopes
                q = ".".join(pref + [n])
                if (path, q) in by_path_qual and by_path_qual[(path, q)] != caller["id"]:
                    return [by_path_qual[(path, q)]], True
                if (path, q) in classes:
                    m = init_of((path, q))
                    return ([m] if m is not None else []), True
                if k == 0:
                    break
            if (path, n) in by_path_qual and by_path_qual[(path, n)] != caller["id"]:
                return [by_path_qual[(path, n)]], True
            imp = tab.get(n)
            if imp and imp[0] == "symbol":
                r = module_def(imp[1], imp[2])
                if r:
                    return r, True
            return [], False
        if isinstance(f, ast.Attribute):
            attr, base = f.attr, f.value
            if isinstance(base, ast.Name) and base.id in ("self", "cls") and caller["cls_self"] is not None:
                m = mro_method((path, caller["cls_self"]), attr)
                if m is not None:
                    return [m], True
            if isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id == "super" and caller["cls_self"] is not None:
                c = classes.get((path, caller["cls_self"]))
                for b in (c or {}).get("bases", []):
                    m = mro_method(class_lookup(path, b), attr)
                    if m is not None:
                        return [m], True
            d = _dotted(base)
            if d:
                head = d.split(".")[0]
                imp = tab.get(head)
                if imp and imp[0] == "module":
                    full = imp[1] + d[len(head):]
                    if full in mod2path:
                        r = module_def(full, attr)
                        if r:
                            return r, True
                ck = class_lookup(path, d)
                if ck is not None:
                    m = mro_method(ck, attr)
                    if m is not None:
                        return [m], True
            return [], False
        return [], False

    adj = [set() for _ in nodes]
    radj = [set() for _ in nodes]
    for caller in nodes:
        col = _Collector()
        for stmt in caller["ast"].body:
            col.visit(stmt)
        for d in caller["ast"].args.defaults + [x for x in caller["ast"].args.kw_defaults if x is not None]:
            col.visit(d)
        for call in col.calls:
            stats["n_calls"] += 1
            tgt, ok = resolve(call, caller)
            if ok and tgt:
                stats["n_direct"] += 1
            else:
                nm = call.func.id if isinstance(call.func, ast.Name) else (call.func.attr if isinstance(call.func, ast.Attribute) else None)
                cands = [j for j in by_name.get(nm, ()) if j != caller["id"]] if nm else []
                if nm and 0 < len(cands) <= FALLBACK_MAX:
                    tgt = cands
                    stats["n_fallback"] += 1
                else:
                    tgt = []
                    stats["n_unresolved"] += 1
            for j in tgt:
                if j != caller["id"]:
                    adj[caller["id"]].add(j)
                    radj[j].add(caller["id"])
    for n in nodes:
        n.pop("ast", None)
    G.nodes, G.adj, G.radj, G.stats = nodes, adj, radj, stats
    G.by_path_qual = by_path_qual
    G.by_name = by_name
    return G


def bfs(adj, sources, limit=None):
    """multi-source BFS distances over adjacency list `adj` -> dict node -> dist"""
    dist = {}
    q = deque()
    for s in sources:
        if s not in dist:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        du = dist[u]
        if limit is not None and du >= limit:
            continue
        for v in adj[u]:
            if v not in dist:
                dist[v] = du + 1
                q.append(v)
    return dist


def undirected(G):
    return [G.adj[i] | G.radj[i] for i in range(len(G.nodes))]


def pagerank(G, keep, d=0.85, iters=60):
    """PageRank on the sub-graph induced by node ids `keep` (list); returns {id: value * N}"""
    ks = set(keep)
    n = len(ks)
    if n == 0:
        return {}
    out = {u: [v for v in G.adj[u] if v in ks] for u in ks}
    pr = {u: 1.0 / n for u in ks}
    for _ in range(iters):
        new = {u: (1.0 - d) / n for u in ks}
        dangling = 0.0
        for u in ks:
            o = out[u]
            if o:
                share = d * pr[u] / len(o)
                for v in o:
                    new[v] += share
            else:
                dangling += pr[u]
        add = d * dangling / n
        pr = {u: new[u] + add for u in ks}
    return {u: pr[u] * n for u in ks}


def ancestors_count(radj, node, within=None):
    seen = {node}
    q = deque([node])
    while q:
        u = q.popleft()
        for v in radj[u]:
            if v not in seen and (within is None or v in within):
                seen.add(v)
                q.append(v)
    return len(seen) - 1
