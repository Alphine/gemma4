
# ----------------------------------------------------------------------------- v8: print the code of the best hit(s)
# After locate the agent's next move was `sed -n 'A,Bp'` on the best hit (one LLM turn).  v8 prints that code inline, line
# numbered, bounded.  SWE_LOCATE_OFF=body switches it off (output == v6); SWE_LOCATE_BODY="n=60,r2=1" overrides the knobs.
BODY = {"n": 60,          # max lines printed for the rank-1 definition
        "k": 2,           # how many of the best definitions get their code printed (1 = rank 1 only)
        "short": 25,      # a further definition is printed only when everything printed so far is at most this many lines
        "n_k": 30,        # max lines for each further definition
        "tot": 80,        # max lines printed over all definitions
        "head": 12,       # lines of a long definition always shown (signature + start)
        "ctx": 3,         # context lines around a best-matching line of a long definition
        "width": 150,     # longer lines are cut
        "total": 4800}    # max characters of the whole output (the harness cuts command output at 5000)
for _kv in os.environ.get("SWE_LOCATE_BODY", "").split(","):
    if "=" in _kv:
        _k, _v = _kv.split("=", 1)
        try:
            BODY[_k.strip()] = int(_v)
        except ValueError:
            pass


def body_on():
    return on("body") and on("v5")


def _span(d):
    """first and last line of the code to show for a definition (a class: header + class-level code up to its first method)"""
    a = d["line"]
    b = d.get("own_end", d["end"]) if d["kind"] == "class" else d["end"]
    return a, max(a, b)


def _line_scores(R, d, a, b, terms, phrase_hits):
    """per line of a..b: how many issue terms it mentions (+ a big bonus for lines that contain text quoted from the issue)"""
    lines = R.file_lines[d["path"]]
    want = []
    for key, (w, tparts, kind, raw) in terms.items():
        want.append((key, [stem(p) for p in tparts], w))
    hits = set(phrase_hits.get(d["path"], ()))
    sc = {}
    for ln in range(a, b + 1):
        text = lines[ln - 1]
        if not text.strip():
            continue
        toks = set()
        for ident in IDENT_RE.findall(text):
            toks.add(ident.lower())
            toks.update(ident_tokens(ident))
        s = 0.0
        for key, tps, w in want:
            if key in toks or (tps and all(t in toks for t in tps)):
                s += w
        if ln in hits:
            s += 5.0
        if s > 0:
            sc[ln] = s
    return sc


def _header_end(R, d, a, b):
    """last line of the signature (a def header may span several lines)"""
    lines = R.file_lines[d["path"]]
    depth = 0
    for ln in range(a, min(b, a + 14) + 1):
        t = lines[ln - 1].split("#")[0]
        depth += t.count("(") + t.count("[") + t.count("{") - t.count(")") - t.count("]") - t.count("}")
        if depth <= 0 and t.rstrip().endswith(":"):
            return ln
    return a


def _merge(segs):
    segs = sorted(segs)
    out = [list(segs[0])]
    for x, y in segs[1:]:
        if x <= out[-1][1] + 2:
            out[-1][1] = max(out[-1][1], y)
        else:
            out.append([x, y])
    return [(x, y) for x, y in out]


def _cost(segs):
    return sum(y - x + 1 for x, y in segs) + len(segs) - 1


def pick_segments(R, d, terms, phrase_hits, n):
    """-> [(start, end)] line ranges to print for definition d, at most n lines.  Short definitions are printed whole; long ones as
    the head (signature + first lines) plus windows around the lines that mention the most issue terms."""
    a, b = _span(d)
    if b - a + 1 <= n:
        return [(a, b)]
    head_end = min(b, a + max(1, int(BODY["head"])) - 1)
    head_end = max(head_end, min(_header_end(R, d, a, b), a + n // 2))
    segs = [(a, head_end)]
    sc = _line_scores(R, d, head_end + 1, b, terms, phrase_hits)
    for ln, s in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0])):
        if any(x <= ln <= y for x, y in segs):
            continue
        for c in (int(BODY["ctx"]), 1, 0):
            cand = _merge(segs + [(max(head_end + 1, ln - c), min(b, ln + c))])
            if _cost(cand) <= n:
                segs = cand
                break
        if _cost(segs) >= n - 1:
            break
    if len(segs) == 1:      # nothing matched: the head plus the lines that follow
        segs = [(a, min(b, a + n - 1))]
    return segs


def _is_stub(R, d):
    """an @overload stub or an empty placeholder: its body is only `...` / `pass`"""
    lines = R.file_lines[d["path"]]
    a, b = _span(d)
    he = _header_end(R, d, a, b)
    body = [l.strip() for l in lines[he:b] if l.strip()]
    if body and all(x in ("...", "pass") for x in body):
        return True
    prev = lines[a - 2].strip() if a >= 2 else ""
    return prev.startswith("@") and "overload" in prev


def expand_twins(R, d):
    """definitions that share (path, class, name) with d - typing overloads, `if version: def f / else: def f` - are folded into one
    ranking entry; print the real implementation(s), not the stub that happened to score best"""
    ids = R.name_idx.get(d["name"], ()) if hasattr(R, "name_idx") else ()
    tw = [R.index[i] for i in ids if R.index[i]["path"] == d["path"] and R.index[i]["cls"] == d["cls"] and R.index[i]["kind"] == d["kind"]]
    if len(tw) <= 1:
        return [d]
    real = [x for x in tw if not _is_stub(R, x)]
    if not real:
        return [d]
    if d in real:
        real = [d] + [x for x in real if x is not d]
    return sorted(real[:2], key=lambda x: x["line"])


def class_methods(final, d0, k):
    """a class hit: its header and attributes say little - print the best ranked method(s) of that class instead (the class itself when it has none)"""
    ms, seen = [], set()
    for s, d, m in final:
        if d["path"] == d0["path"] and d["cls"] == d0["name"] and d["kind"] == "def" and d["name"] not in seen:
            seen.add(d["name"])
            ms.append(d)
            if len(ms) >= k:
                break
    return ms or [d0]


def body_plan(R, top, terms, phrase_hits, n=None, final=None):
    """-> [(rank, d, [(start, end)])] the definitions whose code is printed (n: cap for rank 1, scaled down when the output is too long)"""
    plan, shown, spans = [], 0, []
    n1 = int(n if n is not None else BODY["n"])
    tot = min(int(BODY["tot"]), n1 + int(BODY["tot"]) - int(BODY["n"])) if n is not None else int(BODY["tot"])
    for rank, (s, d0, m) in enumerate(top[:max(1, int(BODY["k"]))], 1):
        base = class_methods(final, d0, 2 if rank == 1 else 1) if (final and on("cls") and d0["kind"] == "class") else [d0]
        for d in [x for b0 in base for x in (expand_twins(R, b0) if on("twins") else [b0])]:
            if not plan:
                n_this = min(n1, tot)
            else:
                if shown > BODY["short"]:
                    break
                n_this = min(int(BODY["n_k"]), tot - shown)
                if n_this < 8:
                    break
            segs = pick_segments(R, d, terms, phrase_hits, n_this)
            a2, b2 = segs[0][0], segs[-1][1]
            if any(d["path"] == pth and a2 <= b1 and a1 <= b2 for pth, a1, b1 in spans):
                continue      # overlaps code that is already shown (a method inside a printed class, ...)
            spans.append((d["path"], a2, b2))
            shown += sum(y - x + 1 for x, y in segs)
            plan.append((rank, d, segs))
    return plan


def render_body(R, plan):
    out = []
    for rank, d, segs in plan:
        lines = R.file_lines[d["path"]]
        a, b = _span(d)
        qual = d.get("qual") or d["name"]
        w = int(BODY["width"])
        out.append("# code of %s:%d-%d (rank %d, %s)" % (d["path"], a, b, rank, qual))
        prev = None
        for x, y in segs:
            if prev is not None and x > prev + 1:
                out.append("      ... lines %d-%d not shown" % (prev + 1, x - 1))
            for ln in range(x, y + 1):
                t = lines[ln - 1].rstrip()
                out.append("%5d: %s" % (ln, t if len(t) <= w else t[:w] + " ..."))
            prev = y
        if prev is not None and prev < b:
            out.append("      ... lines %d-%d not shown" % (prev + 1, b))
    return out
