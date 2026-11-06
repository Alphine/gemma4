"""Build the single-file Kaggle kernel gemma4-ab-extract.py = ../trace_metrics.py + extract_main.py (stdlib only)."""
import ast
from pathlib import Path

here = Path(__file__).resolve().parent
src = (here.parent / "trace_metrics.py").read_text(encoding="utf-8")
main = (here / "extract_main.py").read_text(encoding="utf-8")
# the inlined module must not use "from __future__" after code: keep it first (it is the first statement of trace_metrics.py)
code = src + "\n\n" + main
ast.parse(code)
(here / "gemma4-ab-extract.py").write_text(code, encoding="utf-8")
print("wrote", here / "gemma4-ab-extract.py", len(code), "bytes")
