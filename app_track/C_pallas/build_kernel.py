import json, os, glob
H = os.path.dirname(os.path.abspath(__file__))
files = {"mutate.py": open(os.path.join(H, "mutate.py")).read(), "run_factory.py": open(os.path.join(H, "run_factory.py")).read()}
for p in sorted(glob.glob(os.path.join(H, "seeds", "*.py"))):
    files["seeds/" + os.path.basename(p)] = open(p).read()
script = '''import os, json, subprocess, sys
FILES = json.loads(%r)
os.makedirs("/kaggle/working/src/seeds", exist_ok=True)
for k, v in FILES.items():
    open("/kaggle/working/src/" + k, "w").write(v)
import jax; print("jax", jax.__version__, flush=True)
env = dict(os.environ, NPROC="4")
r = subprocess.run([sys.executable, "/kaggle/working/src/run_factory.py", "/kaggle/working/src/seeds", "/kaggle/working/out", os.environ.get("MAXOP", "8"), "60"], env=env)
print("exit", r.returncode)
''' % json.dumps(files)
d = os.path.join(H, "..", "..", "..", "kernels", "gemma4-app-c-factory")
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "gemma4-app-c-factory.py"), "w").write(script)
open(os.path.join(d, "kernel-metadata.json"), "w").write(json.dumps({
 "id": "<kaggle-owner>/gemma4-app-c-factory", "title": "gemma4-app-c-factory", "code_file": "gemma4-app-c-factory.py",
 "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
 "enable_internet": False, "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": []}, indent=1))
print("built", len(script))
