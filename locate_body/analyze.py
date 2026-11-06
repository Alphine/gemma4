#!/usr/bin/env python3
"""Tables for the v8 body benchmark.  python analyze.py --train rows_synth_train.jsonl --heldout rows_synth_heldout.jsonl --dev body_bench_dev.jsonl --out tables.md"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

CHARS_PER_TOKEN = 3.3
BASE = "r1-60"


def load(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return [r for r in rows if not r.get("skipped") and r.get("gold_lines")]


def ci(vals, n=2000, seed=7):
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    bs = np.sort(a[idx].mean(axis=1))
    return float(a.mean()), float(bs[int(0.025 * n)]), float(bs[int(0.975 * n) - 1])


def pct(c):
    return "%.1f [%.1f, %.1f]" % tuple(100 * x for x in c)


def num(c, d=0):
    return ("%%.%df [%%.%df, %%.%df]" % (d, d, d)) % c


def metric(row, v, key):
    x = row["variants"][v]
    if key == "wrong":          # printed code holds none of the gold lines
        return float(not x["hunk_any"])
    if key == "wrong_samefile":  # ... although the gold file's code is printed (another function of it)
        return float(not x["hunk_any"] and x["gold_file_printed"])
    if key == "wrong_otherfile":
        return float(not x["hunk_any"] and not x["gold_file_printed"])
    if key == "tokens":
        return x["chars"] / CHARS_PER_TOKEN
    if key == "added_tokens":
        return (x["chars"] - row["variants"]["v6"]["chars"]) / CHARS_PER_TOKEN
    if key == "added_chars":
        return x["chars"] - row["variants"]["v6"]["chars"]
    if key == "over4500":
        return float(x["chars"] > 4500)
    return float(x[key])


def table(rows, variants, title, out):
    out.append("")
    out.append("**%s** (n=%d tasks)" % (title, len(rows)))
    out.append("")
    out.append("| variant | edit w/o read: all gold lines printed | any gold line printed | gold function printed | rank-1 is gold fn | WRONG: no gold line printed | of which gold file printed (other fn) | output chars (mean) | added tokens (mean) | >4500 chars | render s (mean) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for v in variants:
        cells = [pct(ci([metric(r, v, k) for r in rows])) for k in ("hunk_all", "hunk_any", "func_shown", "r1_gold_func", "wrong", "wrong_samefile")]
        ch = np.mean([r["variants"][v]["chars"] for r in rows])
        at = np.mean([metric(r, v, "added_tokens") for r in rows])
        o45 = np.mean([metric(r, v, "over4500") for r in rows])
        secs = np.mean([r["variants"][v]["secs"] for r in rows])
        out.append("| %s | %s | %.0f | %.0f | %.1f%% | %.3f |" % (v, " | ".join(cells), ch, at, 100 * o45, secs))


def deltas(rows, variants, base, title, out):
    out.append("")
    out.append("Paired bootstrap, %s (n=%d), difference vs `%s` in percentage points:" % (title, len(rows), base))
    out.append("")
    out.append("| variant | d(all gold lines printed) | d(any gold line printed) | d(WRONG) | d(added tokens) |")
    out.append("|---|---|---|---|---|")
    for v in variants:
        if v == base:
            continue
        cells = []
        for k in ("hunk_all", "hunk_any", "wrong"):
            d = [metric(r, v, k) - metric(r, base, k) for r in rows]
            cells.append("%+.1f [%+.1f, %+.1f]" % tuple(100 * x for x in ci(d)))
        d = [metric(r, v, "added_tokens") - metric(r, base, "added_tokens") for r in rows]
        cells.append("%+.0f [%+.0f, %+.0f]" % ci(d))
        out.append("| %s | %s |" % (v, " | ".join(cells)))


def spans(rows, title, out):
    sp = sorted(max(r["gold_span"]) for r in rows if r["gold_span"])
    hs = sorted(max(r["gold_hunk_span"]) for r in rows if r["gold_hunk_span"])
    q = lambda a, f: a[min(len(a) - 1, int(f * len(a)))]
    out.append("")
    out.append("Gold function length (lines), %s, n=%d: p25 %d, median %d, p75 %d, p90 %d, p95 %d; share <=40 lines %.0f%%, <=60 %.0f%%, <=80 %.0f%%. Gold hunk span (first to last changed line): median %d, p90 %d." % (
        title, len(sp), q(sp, .25), q(sp, .5), q(sp, .75), q(sp, .9), q(sp, .95), 100 * np.mean([x <= 40 for x in sp]),
        100 * np.mean([x <= 60 for x in sp]), 100 * np.mean([x <= 80 for x in sp]), q(hs, .5), q(hs, .9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--heldout")
    ap.add_argument("--dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="")
    a = ap.parse_args()
    out = []
    sets = []
    for name, p in (("synthetic TRAIN", a.train), ("synthetic HELD-OUT", a.heldout), ("DEV (competition tasks)", a.dev)):
        if p and Path(p).exists():
            sets.append((name, load(p)))
    for name, rows in sets:
        variants = [v for v in rows[0]["variants"] if v != "v6"]
        if a.variants:
            variants = a.variants.split(",")
        out.append("\n### " + name)
        spans(rows, name, out)
        table(rows, variants, name + ", all tasks", out)
        deltas(rows, variants, BASE, name, out)
        terse = [r for r in rows if r.get("terse")]
        if terse:
            table(terse, variants, name + ", TERSE tasks only", out)
    Path(a.out).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
