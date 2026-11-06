import ast
import difflib
from collections import Counter
from pathlib import Path

import pytest

from synth.mutators import OPS, Mutation, apply_mutation, find_mutations, mutation_compiles

TOY = (Path(__file__).parent / "toyrepo" / "toypkg" / "text.py").read_text(encoding="utf-8")

RICH = '''"""mod"""
import os

LIMIT = 10
if os.name == "nt":
    LIMIT = 11


class Box:
    def __init__(self, start, end, flag=False):
        self.start = start
        self.end = end
        self.items = []

    def width(self):
        """Width."""
        if self.end < self.start:
            return 0
        return self.end - self.start

    def add(self, x, strict=True):
        try:
            self.items.append(int(x))
        except ValueError:
            if strict:
                raise
            return False
        return True


def clip(s, n=3):
    "Clip."
    if not s or n <= 0:
        return ""
    for i in range(len(s)):
        if s[i] == " ":
            break
    out = s[:n] + s[-1]
    total = 0
    total += n
    return max(out, s.lower().startswith("x") and "y", key=len) if s[0] in "ab" else out


def test_helper_is_ignored(x):
    return x == 1
'''


def changed_lines(old, new):
    return [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]


def test_all_mutations_compile_and_change_code():
    for text in (TOY, RICH):
        muts = find_mutations(text, "pkg/mod.py")
        assert muts
        for m in muts:
            assert mutation_compiles(text, m), (m.op, m.detail)
            new = apply_mutation(text, m)
            assert new != text
            assert ast.dump(ast.parse(new)) != ast.dump(ast.parse(text)), (m.op, m.detail)


def test_operator_coverage_on_rich_example():
    ops = Counter(m.op for m in find_mutations(RICH, "pkg/mod.py"))
    for op in ("cmp_flip", "boolop_flip", "binop_flip", "augassign_flip", "const_off_by_one", "bound_off_by_one",
               "index_flip", "negate_cond", "remove_stmt", "remove_early_exit", "default_value", "return_bool_flip",
               "name_sibling", "attr_sibling", "drop_except", "break_continue"):
        assert ops[op] >= 1, op
    assert set(ops) <= set(OPS)


def test_no_mutations_outside_functions_or_in_tests_and_docstrings():
    muts = find_mutations(RICH, "pkg/mod.py")
    lines = RICH.splitlines()
    for m in muts:
        assert m.func, m
        assert not m.func.split(".")[-1].startswith("test")
        assert "LIMIT" not in lines[m.lineno - 1]
        assert '"""Width."""' not in lines[m.lineno - 1]


def test_minimal_textual_edits():
    muts = find_mutations(RICH, "pkg/mod.py")
    by = {(m.op, m.detail): m for m in muts}
    m = by[("cmp_flip", "< -> <=")]
    assert changed_lines(RICH, apply_mutation(RICH, m)) == ["-        if self.end < self.start:",
                                                            "+        if self.end <= self.start:"]
    m = by[("default_value", "default of strict")]
    assert "+    def add(self, x, strict=False):" in changed_lines(RICH, apply_mutation(RICH, m))
    m = next(x for x in muts if x.op == "drop_except")
    d = changed_lines(RICH, apply_mutation(RICH, m))
    assert d[-1] == "+            raise" and len([l for l in d if l.startswith("-")]) == 3


def test_remove_only_statement_leaves_pass():
    src = "def f(xs):\n    for x in xs:\n        xs.remove(x)\n    return xs\n"
    m = next(m for m in find_mutations(src, "m.py") if m.op == "remove_stmt")
    assert apply_mutation(src, m) == "def f(xs):\n    for x in xs:\n        pass\n    return xs\n"


def test_binop_plus_needs_int_evidence():
    src = "def f(a, b):\n    return a + b\n\ndef g(a):\n    return a + 1\n\ndef h(s):\n    return s + 'x'\n"
    ops = [(m.func, m.op) for m in find_mutations(src, "m.py") if m.op == "binop_flip"]
    assert ops == [("g", "binop_flip")]


def test_multibyte_offsets():
    src = 'def f(s):\n    """ünïcode"""\n    return s.startswith("é") and len(s) > 2\n'
    for m in find_mutations(src, "m.py"):
        assert mutation_compiles(src, m)
    m = next(m for m in find_mutations(src, "m.py") if m.op == "name_sibling")
    assert 'endswith("é")' in apply_mutation(src, m)


def test_json_roundtrip():
    for m in find_mutations(TOY, "toypkg/text.py"):
        m2 = Mutation.from_json(m.to_json())
        assert apply_mutation(TOY, m2) == apply_mutation(TOY, m)
        assert m2.key == m.key


@pytest.mark.parametrize("src", ["def f(:\n", "x = 1\n"])
def test_unparsable_or_empty(src):
    assert find_mutations(src, "m.py") == []
