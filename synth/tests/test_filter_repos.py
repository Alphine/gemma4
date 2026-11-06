import subprocess
import sys
from pathlib import Path

from synth.filter import deny_reason, licence_ok, patch_touches_competition_paths, text_mentions_competition
from synth.repos import REPOS


def test_filter_denies_competition_and_ecosystem():
    for r in ["tiangolo/fastapi", "Textualize/rich", "psf/requests", "encode/httpx", "encode/starlette",
              "encode/uvicorn", "pydantic/pydantic-core", "samuelcolvin/dirty-equals", "agronholm/anyio",
              "someone/fastapi-login", "x/httpx_auth", "y/requests-oauthlib", "z/rich-click", "pypa/pip"]:
        assert deny_reason(r), r
    assert deny_reason("pallets/click", "click")          # shipped in the competition wheels
    assert deny_reason("richardkiss/pycoin") is None       # token rule, not substring
    assert deny_reason("andialbrecht/sqlparse", "sqlparse") is None


def test_candidate_repos_pass_filter_and_are_disjoint():
    pkgs = [p for s in REPOS for p in s["packages"]]
    assert len(pkgs) == len(set(pkgs))
    for s in REPOS:
        for p in s["packages"]:
            assert deny_reason(s["repo"], p) is None, s
        # no candidate's tests may need another candidate's package (all wheels are injected everywhere)
        assert not {d.replace("-", "_").lower() for d in s["test_deps"]} & set(pkgs)


def test_licence_and_text_rules():
    assert licence_ok("MIT") and licence_ok("BSD-3-Clause") and licence_ok("Apache-2.0")
    assert not licence_ok("MPL-2.0") and not licence_ok("GPL-3.0-only") and not licence_ok(None)
    assert text_mentions_competition("see https://github.com/encode/httpx/issues/1")
    assert patch_touches_competition_paths("--- a/src/pkg/_vendor/requests/api.py\n+++ b/src/pkg/_vendor/requests/api.py\n")
    assert not patch_touches_competition_paths("--- a/sqlparse/sql.py\n+++ b/sqlparse/sql.py\n")


@__import__("pytest").mark.skip(reason="needs the repository kernels/ layout (checks generated kernels against sources)")
def test_generated_kernels_up_to_date():
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run([sys.executable, str(root / "synth" / "build_kernels.py"), "--check"], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout


def test_all_modules_import():
    import importlib
    for m in ("common", "factory", "validate", "statements", "mutators", "patches", "filter", "repos"):
        importlib.import_module(f"synth.{m}")
