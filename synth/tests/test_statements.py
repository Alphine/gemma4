import re
from collections import Counter

import pytest

from synth.mutators import apply_mutation, find_mutations
from synth.statements import (FailureView, build_repro, generate, humanize_test_name, mutated_modules,
                              strip_module)

from .helpers import copy_toy, run_plugin

SCRUB = ["toypkg/text.py"]


def failures_for(tmp_path, pred):
    repo = copy_toy(tmp_path)
    p = repo / "toypkg" / "text.py"
    text = p.read_text(encoding="utf-8")
    m = next(m for m in find_mutations(text, "toypkg/text.py") if pred(m))
    p.write_bytes(apply_mutation(text, m).encode("utf-8"))
    _, data = run_plugin(repo, ["tests"])
    test_src = (repo / "tests" / "test_text.py").read_text(encoding="utf-8")
    fails = []
    for nodeid, rec in sorted(data["tests"].items()):
        if rec["outcome"] == "failed":
            fails.append(dict(rec, nodeid=nodeid, test_src=test_src))
    return fails, str(repo).replace("\\", "/")


@pytest.fixture(scope="module")
def cmp_fail(tmp_path_factory):
    return failures_for(tmp_path_factory.mktemp("a"), lambda m: m.op == "cmp_flip" and m.detail == ">= -> >")


@pytest.fixture(scope="module")
def exc_fail(tmp_path_factory):
    # dropping `self.end = end` makes width() raise AttributeError
    return failures_for(tmp_path_factory.mktemp("b"),
                        lambda m: m.op == "remove_stmt" and m.func == "Span.__init__" and m.lineno == 26)


def no_leak(text):
    assert "text.py" not in text and "toypkg/text" not in text and "toypkg.text" not in text
    assert not re.search(r"line \d+, in (count_words|width|clip|contains)", text)
    assert ">=" not in text or "assert" in text   # the operator only appears inside quoted test code


@pytest.mark.parametrize("style", ["terse", "observed", "repro"])
def test_styles_render_without_leaks(cmp_fail, style):
    fails, root = cmp_fail
    text, got, info = generate("toy_s0001", fails, ["toypkg"], SCRUB, roots=[root], force_style=style)
    assert got == style
    no_leak(text)
    assert root not in text
    if style == "terse":
        assert "\n" not in text.strip()
        assert "count_words" in text
    if style == "observed":
        assert "`2`" in text or "`1`" in text or "`0`" in text
    if style == "repro":
        code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
        compile(code, "repro.py", "exec")
        assert "from toypkg import count_words" in code
        assert "WORDS = " in code and "n = " in code
        assert "Traceback (most recent call last):" in text and "AssertionError" in text


def test_exception_failure_statements(exc_fail):
    fails, root = exc_fail
    assert fails and all(f["exc_type"] == "AttributeError" for f in fails)
    for style in ("terse", "observed", "repro"):
        text, got, _ = generate("toy_s0002", fails, ["toypkg"], SCRUB, roots=[root], force_style=style)
        no_leak(text)
        if got != "terse":
            assert "AttributeError" in text


def test_deterministic_and_style_mix(cmp_fail):
    fails, root = cmp_fail
    a = generate("toy_s0003", fails, ["toypkg"], SCRUB, roots=[root])
    b = generate("toy_s0003", fails, ["toypkg"], SCRUB, roots=[root])
    assert a == b
    styles = Counter(generate(f"toy_{i}", fails, ["toypkg"], SCRUB, roots=[root])[1] for i in range(300))
    assert 0.28 <= styles["terse"] / 300 <= 0.45, styles
    assert styles["repro"] > 50 and styles["observed"] > 50


def test_helpers():
    assert humanize_test_name("test_contains_end_excluded") == "contains end excluded"
    assert humanize_test_name("test_2", ("TestSpanWidth",)) == "span width"
    assert mutated_modules(["src/pkg/sub/mod.py", "pkg/__init__.py", "single.py"]) == ["pkg.sub.mod"]


def test_unittest_style_repro_conversion():
    src = ("import unittest\nfrom toypkg import clip\n\n\nclass TestClip(unittest.TestCase):\n"
           "    def test_basic(self):\n        self.assertEqual(clip('abc', 2), 'ab')\n")
    fail = {"nodeid": "tests/test_u.py::TestClip::test_basic", "test_src": src, "exc_type": "AssertionError",
            "exc_msg": "'a' != 'ab'", "tb": [["/x/tests/test_u.py", 7, "test_basic"]]}
    v = FailureView(fail, ["toypkg"], SCRUB)
    snippet, line, shown = build_repro(v)
    assert "assert clip('abc', 2) == 'ab'" in snippet and shown.startswith("assert clip")
    assert v.observed_expected() == ("'a'", "'ab'")


def test_strip_module():
    assert strip_module("class_registry.get_class", ["src/marshmallow/class_registry.py"]) == "get_class"
    assert strip_module("marshmallow.class_registry.get_class", ["src/marshmallow/class_registry.py"]) == "get_class"
    assert strip_module("sqlparse.format", ["sqlparse/formatter.py"]) == "sqlparse.format"
    assert strip_module("parse.parse", ["parse/__init__.py"]) == "parse.parse"


def test_pair_groups_describe_both(cmp_fail, exc_fail):
    f1, root1 = cmp_fail
    f2, root2 = exc_fail
    fails = f1 + f2
    groups = [[f["nodeid"] for f in f1], [f["nodeid"] for f in f2]]
    for style in ("terse", "observed", "repro"):
        text, got, info = generate("toy_pair", fails, ["toypkg"], SCRUB, roots=[root1, root2], force_style=style,
                                   groups=groups)
        assert "count_words" in text and "Span" in text, text
        assert info["n_groups"] == 2
        no_leak(text)


def test_unittest_list_diff_prefix():
    from synth.statements import postprocess_statement
    fail = {"nodeid": "tests/test_u.py::TestClip::test_basic", "exc_type": "AssertionError",
            "test_src": ("from toypkg import clip\n\nclass TestClip:\n    def test_basic(self):\n"
                         "        self.assertEqual(clip([1], 2), [1, 2])\n"),
            "exc_msg": "Lists differ: [1] != [1, 2]\n\nSecond list contains 1 additional elements.",
            "tb": [["/x/tests/test_u.py", 5, "test_basic"]]}
    assert FailureView(fail, ["toypkg"], SCRUB).observed_expected() == ("[1]", "[1, 2]")
    assert postprocess_statement("got `Lists differ: [1]` instead") == "got `[1]` instead"
