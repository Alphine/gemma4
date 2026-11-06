"""Candidate source repos for synthetic tasks.

Selection rules (see synth/README.md): pure Python, permissive licence (checked at build time via the GitHub
API and recorded), fast hermetic pytest suite, few dependencies, passes synth.filter.deny_reason (not a
competition repo or its ecosystem, package not shipped in the competition wheels), and no task repo is a
dependency of another candidate's tests (every grader sandbox gets *all* wheels of the wheel dir injected,
so a task repo's own package must never be in it).

Fields: repo (GitHub owner/name), packages (import names), test_deps (extra pip names needed by the tests;
runtime deps from pyproject are added automatically), tag (None = latest stable release tag).
"""

REPOS = [
    {"repo": "andialbrecht/sqlparse", "packages": ["sqlparse"], "test_deps": []},
    {"repo": "mahmoud/boltons", "packages": ["boltons"], "test_deps": []},
    {"repo": "lepture/mistune", "packages": ["mistune"], "test_deps": []},
    {"repo": "pyparsing/pyparsing", "packages": ["pyparsing"], "test_deps": ["railroad-diagrams"]},
    {"repo": "pytoolz/toolz", "packages": ["toolz", "tlz"], "test_deps": []},
    # more-itertools: dropped (test_more.py has a >60 s test, whole-file runs exceed the budget)
    {"repo": "python-poetry/tomlkit", "packages": ["tomlkit"], "test_deps": []},
    {"repo": "r1chardj0n3s/parse", "packages": ["parse"], "test_deps": []},
    {"repo": "tkem/cachetools", "packages": ["cachetools"], "test_deps": []},
    {"repo": "jpvanhal/inflection", "packages": ["inflection"], "test_deps": []},
    {"repo": "jmespath/jmespath.py", "packages": ["jmespath"], "test_deps": []},
    {"repo": "astanin/python-tabulate", "packages": ["tabulate"], "test_deps": ["wcwidth"]},
    {"repo": "martinblech/xmltodict", "packages": ["xmltodict"], "test_deps": []},
    {"repo": "PyCQA/pyflakes", "packages": ["pyflakes"], "test_deps": []},
    {"repo": "Python-Markdown/markdown", "packages": ["markdown"], "test_deps": []},
    {"repo": "python-semver/python-semver", "packages": ["semver"], "test_deps": []},
    {"repo": "marshmallow-code/marshmallow", "packages": ["marshmallow"], "test_deps": ["simplejson"]},
    {"repo": "msiemens/tinydb", "packages": ["tinydb"], "test_deps": []},
    {"repo": "dgilland/pydash", "packages": ["pydash"], "test_deps": []},
    {"repo": "hukkin/tomli", "packages": ["tomli"], "test_deps": []},
    {"repo": "dbader/schedule", "packages": ["schedule"], "test_deps": []},
    {"repo": "scrapy/w3lib", "packages": ["w3lib"], "test_deps": ["hypothesis"]},
    {"repo": "PyCQA/pycodestyle", "packages": ["pycodestyle"], "test_deps": []},
    {"repo": "stefankoegl/python-json-pointer", "packages": ["jsonpointer"], "test_deps": []},
    {"repo": "Suor/funcy", "packages": ["funcy"], "test_deps": []},
    {"repo": "alecthomas/voluptuous", "packages": ["voluptuous"], "test_deps": []},
]

# never install these from a repo's declared deps (tooling, not needed to run the tests, or provided by the
# grader image); pytest plugins that only add reporting are dropped too
SKIP_DEPS = {
    "pytest", "pytest-cov", "coverage", "tox", "pre-commit", "mypy", "flake8", "ruff", "black", "isort",
    "sphinx", "sphinx-rtd-theme", "furo", "codecov", "pylint", "build", "twine", "wheel", "setuptools", "pip",
    "pytest-xdist", "pytest-timeout", "pytest-randomly", "pytest-benchmark", "pytest-codspeed", "nox",
    "types-setuptools", "typing-extensions", "check-manifest", "bump2version", "readme-renderer", "pyright",
    "pytest-sugar", "pytest-instafail", "pytest-rerunfailures", "pytest-env", "docutils", "mkdocs",
    "pytest-mypy-plugins", "tomli", "importlib-metadata", "zipp", "exceptiongroup", "typing", "mypy-extensions",
}
