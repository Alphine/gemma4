import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SYNTH = Path(__file__).resolve().parents[1]
TOY = Path(__file__).resolve().parent / "toyrepo"


def copy_toy(tmp_path):
    dst = Path(tmp_path) / "toyrepo"
    shutil.copytree(TOY, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return dst


def run_plugin(repo, args, trace=()):
    out = Path(repo) / "_synth_out.json"
    env = dict(os.environ)
    env.update({"PYTHONPATH": os.pathsep.join([str(SYNTH), str(repo)]), "SYNTH_OUT": str(out),
                "SYNTH_TRACE": os.pathsep.join(str(Path(repo) / t) for t in trace)})
    r = subprocess.run([sys.executable, "-m", "pytest", "-p", "synth_plug", "-p", "no:cacheprovider", "-q", *args],
                       cwd=repo, env=env, capture_output=True, text=True, timeout=120)
    data = json.loads(out.read_text(encoding="utf-8"))
    out.unlink()
    return r, data
