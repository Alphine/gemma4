import ast
import shutil
import subprocess
from pathlib import Path

import pytest

from synth.mutators import apply_mutation, find_mutations
from synth.patches import make_patch, parse_nodeid, remove_tests, unified_diff_git

from .helpers import copy_toy, run_plugin


def test_parse_nodeid():
    assert parse_nodeid("tests/t.py::A::B::test_x[p-1]") == ("tests/t.py", ("A", "B"), "test_x", "p-1")
    assert parse_nodeid("tests/t.py::test_y") == ("tests/t.py", (), "test_y", None)


def test_plugin_records_outcomes_and_coverage(tmp_path):
    repo = copy_toy(tmp_path)
    r, data = run_plugin(repo, ["tests"], trace=["toypkg"])
    assert r.returncode == 0, r.stdout + r.stderr
    tests = data["tests"]
    assert len(tests) == 8 and all(t["outcome"] == "passed" for t in tests.values())
    cov = data["cover"]["tests/test_text.py::TestSpan::test_width"]
    (fname, lines), = [(k, v) for k, v in cov.items() if k.endswith("text.py")]
    src = (repo / "toypkg" / "text.py").read_text().splitlines()
    assert any("return self.end - self.start" in src[l - 1] for l in lines)
    assert tests["tests/test_text.py::test_count_words_min_len[1-3]"]["params"] == {"n": "1", "expected": "3"}


def _mutate(repo, pred):
    p = repo / "toypkg" / "text.py"
    text = p.read_text(encoding="utf-8")
    m = next(m for m in find_mutations(text, "toypkg/text.py") if pred(m, text))
    p.write_text(apply_mutation(text, m), encoding="utf-8")
    return text, m


def test_mutation_failures_recorded(tmp_path):
    repo = copy_toy(tmp_path)
    _mutate(repo, lambda m, t: m.op == "cmp_flip" and m.detail == ">= -> >")
    r, data = run_plugin(repo, ["tests"])
    failed = sorted(k for k, v in data["tests"].items() if v["outcome"] == "failed")
    assert failed == [f"tests/test_text.py::test_count_words_min_len[{p}]" for p in ("1-3", "2-2", "3-1")]
    rec = data["tests"][failed[0]]
    assert rec["exc_type"] == "AssertionError" and rec["exc_msg"].startswith("assert 2 == 3")
    assert any(fn == "test_count_words_min_len" for _, _, fn in rec["tb"])


def test_remove_tests_and_roundtrip_with_git(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    repo = copy_toy(tmp_path)
    tf = repo / "tests" / "test_text.py"
    orig_test = tf.read_text(encoding="utf-8")
    new_test, removed = remove_tests(orig_test, [((), "test_count_words_min_len"), (("TestSpan",), "test_width")])
    assert "test_count_words_min_len" not in new_test and "def test_width" not in new_test
    assert "test_contains_end_excluded" in new_test and "test_clip_basic" in new_test
    ast.parse(new_test)
    # class emptied entirely -> pass
    only, _ = remove_tests("class TestA:\n    def test_a(self):\n        pass\n", [(("TestA",), "test_a")])
    assert only == "class TestA:\n    pass\n"

    src_p = repo / "toypkg" / "text.py"
    orig_src = src_p.read_text(encoding="utf-8")
    m = next(m for m in find_mutations(orig_src, "toypkg/text.py") if m.op == "cmp_flip" and m.detail == ">= -> >")
    bug_src = apply_mutation(orig_src, m)
    start_patch = make_patch({"toypkg/text.py": (orig_src, bug_src), "tests/test_text.py": (orig_test, new_test)})
    gold = unified_diff_git("toypkg/text.py", bug_src, orig_src)
    test_patch = unified_diff_git("tests/test_text.py", new_test, orig_test)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    for name, patch in (("start", start_patch), ("test", test_patch), ("gold", gold)):
        pf = tmp_path / f"{name}.patch"
        pf.write_bytes(patch.encode("utf-8"))
        r = subprocess.run(["git", "apply", str(pf)], cwd=repo, capture_output=True, text=True)
        assert r.returncode == 0, (name, r.stderr)
    assert src_p.read_text(encoding="utf-8") == orig_src
    assert tf.read_text(encoding="utf-8") == orig_test


def test_diff_no_trailing_newline():
    d = unified_diff_git("a.py", "x = 1\ny = 2", "x = 1\ny = 3")
    assert d.endswith("+y = 3\n\\ No newline at end of file\n")
    assert unified_diff_git("a.py", "same\n", "same\n") == ""
