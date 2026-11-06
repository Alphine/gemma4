"""Contamination + licence filter for synthetic-task source repos (research/data-scaling/README.md, section 2).

A repo may be used only if it is not a competition repo or part of its ecosystem and has a permissive licence.
"""
from __future__ import annotations

import re

# hard tier: the competition repos themselves (and their old homes)
HARD_DENY = {
    "tiangolo/fastapi", "fastapi/fastapi", "textualize/rich", "psf/requests", "kennethreitz/requests",
    "requests/requests", "encode/httpx",
}
# soft tier: their ecosystem
SOFT_DENY_REPOS = {
    "urllib3/urllib3", "agronholm/anyio", "python-hyper/h11", "pypa/pip", "encode/starlette",
    "encode/httpcore", "pydantic/pydantic", "pydantic/pydantic-core", "samuelcolvin/pydantic",
}
SOFT_DENY_OWNERS = {"encode", "textualize", "tiangolo", "fastapi", "pydantic", "samuelcolvin"}
# any [-_./]-separated token of owner/name equal to one of these drops the repo
DENY_TOKENS = {"fastapi", "rich", "requests", "httpx", "starlette", "httpcore", "h11", "anyio", "urllib3",
               "pydantic", "pip"}
# python packages that must not be a task repo's package (installed from the competition wheels into every
# grader sandbox: a workspace copy would be shadowed by the wheel, so the gold patch could not take effect)
COMPETITION_WHEEL_PACKAGES = {
    "flask", "annotated_types", "anyio", "certifi", "charset_normalizer", "click", "colorama", "editables",
    "fastapi", "flit_core", "hatchling", "httpcore", "httpx", "idna", "iniconfig", "itsdangerous", "jinja2",
    "markdown_it_py", "markupsafe", "mdurl", "packaging", "pdm_backend", "pluggy", "poetry_core", "pydantic",
    "pydantic_core", "pygments", "pytest", "requests", "rich", "setuptools", "sniffio", "sqlalchemy", "sqlmodel",
    "starlette", "tqdm", "typer", "typing_extensions", "urllib3", "werkzeug", "wheel",
    # image packages / host extras copied into the emulated image
    "pip", "pytest_timeout", "shellingham", "pathspec", "trove_classifiers",
}
PERMISSIVE = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "PSF-2.0", "Python-2.0", "0BSD",
              "BSD-3-Clause-Clear", "MIT-0"}


def norm_repo(repo: str) -> str:
    return repo.strip().lower().rstrip("/").removesuffix(".git").split("github.com/")[-1]


def repo_tokens(repo: str) -> set[str]:
    return {t for t in re.split(r"[-_./]+", norm_repo(repo)) if t}


def deny_reason(repo: str, package: str | None = None) -> str | None:
    """Why `repo` (owner/name) must not be used, or None if it passes the contamination filter."""
    r = norm_repo(repo)
    if r in HARD_DENY:
        return "hard-tier competition repo"
    if r in SOFT_DENY_REPOS:
        return "soft-tier ecosystem repo"
    owner = r.split("/", 1)[0]
    if owner in SOFT_DENY_OWNERS:
        return f"soft-tier owner {owner}/*"
    bad = repo_tokens(r) & DENY_TOKENS
    if bad:
        return f"name token {sorted(bad)[0]!r}"
    if package and re.sub(r"[-.]+", "_", package.lower()) in COMPETITION_WHEEL_PACKAGES:
        return f"package {package!r} ships in the competition wheels (would shadow the workspace)"
    return None


def licence_ok(spdx: str | None) -> bool:
    return bool(spdx) and spdx in PERMISSIVE


def text_mentions_competition(text: str) -> bool:
    """Link rule: a statement/patch must not point at a competition repo."""
    return bool(re.search(r"github\.com/(tiangolo|fastapi)/fastapi|github\.com/textualize/rich|"
                          r"github\.com/psf/requests|github\.com/encode/httpx", text or "", re.I))


def patch_touches_competition_paths(patch: str) -> bool:
    """Path rule: vendored copies of the competition packages."""
    paths = re.findall(r"^(?:\+\+\+|---) [ab]/(\S+)", patch or "", re.M)
    return any(re.search(r"(^|/)(_vendor/|vendor/)?(fastapi|rich|requests|httpx)/", p) for p in paths)
