"""git-apply compatible diffs and hidden-test removal for synthetic tasks.

Task layout (competition schema, dedup snapshot form):
  snapshots/base_snapshots/base_<repo>.tgz   pristine repo at the pinned commit (fresh `git init`, no commits)
  snapshots/patches/<iid>.patch               base -> task start state: the bug + the failing tests removed
  patch       (gold)                          task start -> fixed source (reverse of the mutation)
  test_patch                                  re-adds the removed failing tests (verify_task runs these files)
"""
from __future__ import annotations

import ast
import difflib
import re


def parse_nodeid(nodeid: str):
    """'tests/t.py::A::B::test_x[p-1]' -> ('tests/t.py', ('A', 'B'), 'test_x', 'p-1')"""
    path, _, rest = nodeid.partition("::")
    parts = rest.split("::") if rest else []
    if not parts:
        return path, (), "", None
    last = parts[-1]
    param = None
    m = re.match(r"^([^\[]+)\[(.*)\]$", last)
    if m:
        last, param = m.group(1), m.group(2)
    return path, tuple(parts[:-1]), last, param


def unified_diff_git(path: str, old: str, new: str, context: int = 3) -> str:
    if old == new:
        return ""
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    out = [f"diff --git a/{path} b/{path}\n"]
    lines = list(difflib.unified_diff(a, b, fromfile=f"a/{path}", tofile=f"b/{path}", n=context))
    for ln in lines:
        if ln.startswith(("---", "+++")):
            out.append(ln.rstrip("\n") + "\n")
        elif ln.endswith("\n"):
            out.append(ln)
        else:   # last line without newline
            out.append(ln + "\n\\ No newline at end of file\n")
    return "".join(out)


def make_patch(changes: dict) -> str:
    """changes: {relpath: (old_text, new_text)} -> one patch (files sorted)."""
    return "".join(unified_diff_git(p, o, n) for p, (o, n) in sorted(changes.items()))


def _def_span(node):
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return first, node.end_lineno


def find_test_def(tree, classes, name):
    """The FunctionDef for Class::...::name (or a module-level def), else None."""
    body = tree.body
    for c in classes:
        cls = next((n for n in body if isinstance(n, ast.ClassDef) and n.name == c), None)
        if cls is None:
            return None, None
        body = cls.body
    fn = next((n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
    return fn, body


def remove_tests(src: str, targets) -> tuple[str, list]:
    """Remove the test functions `targets` [(classes_tuple, name), ...] from a test module source.

    Returns (new_src, removed) where removed lists the (classes, name) actually removed. A class body left
    empty gets a `pass`. Raises ValueError if a target cannot be found.
    """
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    spans, removed, bodies = [], [], []
    for classes, name in sorted(set(targets)):
        fn, body = find_test_def(tree, classes, name)
        if fn is None:
            raise ValueError(f"test {'::'.join(classes + (name,))} not found")
        spans.append((_def_span(fn), fn, body))
        removed.append((classes, name))
    drop = set()
    for (first, last), fn, body in spans:
        for i in range(first, last + 1):
            drop.add(i)
        # also drop the blank lines that separated it from the previous statement (keeps spacing tidy)
        j = first - 1
        while j >= 1 and lines[j - 1].strip() == "" and j not in drop:
            drop.add(j)
            j -= 1
    # empty class bodies -> pass
    add_pass = {}
    for (first, last), fn, body in spans:
        remaining = [n for n in body if not (n.lineno in drop and n.end_lineno in drop)]
        meaningful = [n for n in remaining if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None),
                                                                                           ast.Constant))]
        if not meaningful:
            indent = re.match(r"\s*", lines[fn.lineno - 1]).group(0)
            add_pass[first] = indent + "pass\n"
    out = []
    for i, ln in enumerate(lines, start=1):
        if i in add_pass:
            out.append(add_pass[i])
        if i not in drop:
            out.append(ln)
    new = "".join(out)
    # keep two blank lines before following top-level defs where a removal collapsed them
    new = re.sub(r"\n{4,}", "\n\n\n", new)
    ast.parse(new)   # must still be valid python
    return new, removed


def patch_files(patch: str) -> list:
    return re.findall(r"^\+\+\+ b/(\S+)", patch, re.M)
