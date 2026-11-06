"""Kernel-side shared helpers: logging, shell, harness install, merged data dir, emulated sandboxes.

The grader-image emulation itself is reused from kernels/gemma4-lb-eval/gemma4-lb-eval.py (embedded into the
synth kernels as `gemma4_lb_eval`): ContainerManager (SubprocessManager subclass on python3.13, no host
site-packages, image packages pre-seeded, harness docker code path), sitecustomize neutralised, /wheels.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_OUT", "/kaggle/working"))
SCRATCH = Path(os.environ.get("SYNTH_SCRATCH", "/root/synth"))   # no "/tmp" or "/workspace" token in it
_LOGF = None


def log(msg):
    global _LOGF
    line = f"[{(time.time() - T0) / 60:6.1f}m] {msg}"
    print(line, flush=True)
    try:
        if _LOGF is None:
            WORK.mkdir(parents=True, exist_ok=True)
            _LOGF = open(WORK / "synth.log", "a", encoding="utf-8")
        _LOGF.write(line + "\n")
        _LOGF.flush()
    except OSError:
        pass


def minutes():
    return (time.time() - T0) / 60


def sh(cmd, cwd=None, timeout=600, check=False, env=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, capture_output=True, text=True,
                       timeout=timeout, env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r


def find_all(pattern, prefer=None, want_dir=False):
    hits = [Path(p) for p in glob.glob(str(INPUT / "**" / pattern), recursive=True)]
    hits = [h for h in hits if h.is_dir() == want_dir]
    hits.sort(key=lambda p: ((0 if (prefer and prefer in str(p)) else 1), len(str(p))))
    return hits


def find_one(pattern, prefer=None, want_dir=False):
    hits = find_all(pattern, prefer, want_dir)
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def comp_data_dir():
    return find_one("tasks.jsonl", prefer="/competitions/").parent


def install_harness():
    """swegemma + its pure deps from the wheelhouse (no vLLM / CUDA stack), as kernels/gemma4-gold-control."""
    os.environ.update({"LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False",
                       "OTEL_SDK_DISABLED": "true", "TRANSFORMERS_NO_TF": "1"})
    wh = find_one("swegemma-*.whl", prefer="wheelhouse").parent
    skip = ("vllm", "flashinfer", "nvidia", "quack", "bitsandbytes", "transformers", "torch", "triton", "xformers")
    whl = [str(w) for w in wh.glob("*.whl")
           if "cp312" not in w.name and "cutlass" not in w.name.lower() and not w.name.lower().startswith(skip)]
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *whl], capture_output=True,
                       text=True)
    import importlib
    importlib.invalidate_caches()
    return r.returncode


def lb_module():
    """The embedded gemma4-lb-eval kernel as a module (grader-image emulation)."""
    import gemma4_lb_eval as LB   # noqa: written next to the synth package by the kernel bootstrap
    LB.log = log
    return LB


def build_merged_data_dir(dst: Path, comp: Path, tasks_jsonl: Path | None, snapshots: Path, extra_wheels: Path):
    """A data dir the harness can use like the competition's: our tasks + snapshots, the competition's
    sandbox/setup.py + docker shims, and wheels = competition wheels + our extra pure-python wheels."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    if tasks_jsonl is not None:
        os.symlink(tasks_jsonl, dst / "tasks.jsonl")
    else:
        (dst / "tasks.jsonl").write_text("")
    os.symlink(snapshots, dst / "snapshots")
    for sub in ("sandbox", "docker"):
        os.symlink(comp / sub, dst / sub)
    w = dst / "wheels"
    w.mkdir()
    for f in (comp / "wheels").iterdir():
        os.symlink(f, w / f.name)
    if extra_wheels and extra_wheels.exists():
        for f in extra_wheels.glob("*.whl"):
            if not (w / f.name).exists():
                os.symlink(f, w / f.name)
    return dst


def make_manager(LB, timeout_seconds=600):
    """The lb-eval emulated ContainerManager, plus a `pip` shim in each sandbox venv when the venv was created
    without pip (no ensurepip on the Kaggle host). Without it the harness's `pip install -e /workspace` falls
    through to the HOST pip: the repo is editable-installed into the host interpreter (replacing e.g. the host's
    mistune/pyparsing, which later breaks nbconvert) and never into the sandbox, unlike the real grader image."""
    base = LB.make_manager_cls()

    class SynthContainerManager(base):
        shims = 0

        def start(self):
            sid = super().start()
            venv = Path(self.sandboxes[sid]["venv"])
            for name in ("pip", "pip3", "pip3.13"):
                p = venv / "bin" / name
                if not p.exists():
                    p.write_text(f'#!/bin/sh\nexec "{venv}/bin/python3" -m pip "$@"\n')
                    p.chmod(0o755)
                    type(self).shims += 1
            return sid

    SynthContainerManager.__name__ = "ContainerManager"     # the harness keys its docker path on this name
    SynthContainerManager.__qualname__ = "ContainerManager"
    return SynthContainerManager(timeout_seconds=timeout_seconds)


def host_import_ok(mod):
    r = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True, text=True)
    return r.returncode == 0


def make_cfg(data_dir: Path, results: Path, timeout=300):
    from swegemma.config import EvalConfig
    sub = results / "dummy_submission"
    sub.mkdir(parents=True, exist_ok=True)
    cfg = EvalConfig(tasks_path=data_dir / "tasks.jsonl", snapshots_dir=data_dir / "snapshots",
                     results_dir=results, submission_dir=sub, models=None, sandbox="docker",
                     wheels_dir=data_dir / "wheels", display_mode="quiet")
    return cfg


class Sandbox:
    """One emulated grader sandbox prepared exactly like verify_task does before running tests."""

    def __init__(self, mgr, cfg, repo, snapshot, base=None, patch=None, agent=False):
        """agent=True: stop where the harness's agent sandbox (Phase 1) stops, after the baseline commit."""
        from swegemma.harness import container_setup as cs
        self.mgr, self.cfg, self.repo = mgr, cfg, repo
        self.sid = mgr.start()
        cs.setup_container_wheels(mgr, self.sid, cfg)
        cs.extract_snapshot(mgr, self.sid, snapshot, patch_path=patch, base_snapshot_path=base)
        cs.setup_synthetic_hardware(mgr, self.sid)
        cs.setup_git_exclude(mgr, self.sid)
        cs.install_editable_package(mgr, self.sid)
        cs.install_test_dependencies(mgr, self.sid, repo, fast_path=True, config=cfg)
        cs.setup_workspace_test_config(mgr, self.sid, repo=repo)
        cs.setup_baseline_commit(mgr, self.sid, "baseline" if agent else "eval_baseline")
        self.ws = Path(mgr.sandboxes[self.sid]["workspace"])
        if agent:
            return
        cs.install_test_dependencies(mgr, self.sid, repo, fast_path=True, config=cfg)
        cs.setup_workspace_test_config(mgr, self.sid, repo=repo, overwrite=True)
        self.ws = Path(mgr.sandboxes[self.sid]["workspace"])

    def exec(self, cmd, timeout=300):
        return self.mgr.exec(self.sid, cmd, timeout=timeout)

    def pytest(self, targets, out_json: Path, plug_dir: Path, trace=(), alarm=0, timeout=300):
        """Run pytest like verify_task (same flags/env) plus the synth plugin. -> (exit, wall_s, data|None)"""
        import shlex
        out_json.parent.mkdir(parents=True, exist_ok=True)
        if out_json.exists():
            out_json.unlink()
        args = " ".join(shlex.quote(t) for t in targets)
        env = (f"PYTHONPATH={shlex.quote(str(plug_dir))} SYNTH_OUT={shlex.quote(str(out_json))} "
               f"SYNTH_TRACE={shlex.quote(os.pathsep.join(trace))} SYNTH_ALARM={int(alarm)} "
               "PYTHONDONTWRITEBYTECODE=1 ")
        cmd = (f"cd /workspace && {env}PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -m pytest {args} "
               "-p synth_plug -p no:cacheprovider -p no:anyio -o timeout=0 -o python_classes=\"Test* *Test\" -q")
        t = time.time()
        r = self.exec(cmd, timeout=timeout)
        wall = time.time() - t
        data = None
        if out_json.exists():
            try:
                data = json.loads(out_json.read_text(encoding="utf-8"))
            except Exception:
                data = None
        return r, wall, data

    def stop(self):
        try:
            self.mgr.stop(self.sid)
        except Exception:
            pass


def norm(name):
    return re.sub(r"[-_.]+", "_", name).lower()
