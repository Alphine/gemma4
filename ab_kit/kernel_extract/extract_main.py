# ---- main of the Kaggle CPU extraction kernel (appended after the inlined trace_metrics.py by build_kernel.py) ----
import json as _json
import os as _os
import sys as _sys
import tarfile as _tarfile
import time as _time
from pathlib import Path as _Path

_IN = _Path(_os.environ.get("AB_IN", "/kaggle/input"))
_OUT = _Path(_os.environ.get("AB_OUT", "/kaggle/working"))
_TAG = _os.environ.get("AB_TAG", _time.strftime("%Y%m%d-%H%M%S"))


def _arm_task(name):
    parts = [x for x in name.replace(chr(92), "/").split("/") if x]
    stem = parts[-1][:-5]
    if stem.startswith("trace_"):
        stem = stem[6:]
    return (parts[-2] if len(parts) > 1 else "?"), stem


def _blobs():
    for p in sorted(_IN.rglob("*")):
        if p.is_file() and p.name.endswith((".tar", ".tar.gz", ".tgz")):
            with _tarfile.open(p, "r:*") as t:
                for m in t:
                    if m.isfile() and m.name.endswith(".json") and m.size < 80_000_000:
                        yield m.name, t.extractfile(m).read()
        elif p.is_file() and p.suffix == ".json" and "traces" in p.parts:
            yield str(p.relative_to(_IN)), p.read_bytes()


def main():
    rows, n_bad = [], 0
    for name, blob in _blobs():
        arm, task = _arm_task(name)
        m = parse_trace_text(blob.decode("utf-8", "replace"))
        n_bad += not m.get("trace_ok")
        rows.append({"arm": arm, "instance_id": task, "metrics": m, "source": name})
    _OUT.mkdir(parents=True, exist_ok=True)
    out = _OUT / f"trace_metrics_{_TAG}.jsonl"
    out.write_text("\n".join(_json.dumps(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"wrote {out} rows={len(rows)} unreadable={n_bad} bytes={out.stat().st_size}")
    return 0 if rows else 1


if __name__ == "__main__":
    _sys.exit(main())
