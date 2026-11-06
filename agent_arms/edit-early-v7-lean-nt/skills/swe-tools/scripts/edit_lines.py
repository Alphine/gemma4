"""edit_lines.py - replace a line range of a file (no old_string matching, no escaping problems).

Usage (run_skill_script): args {"path": "pkg/mod.py", "start": 120, "end": 124,
                                "expect": "<text that is inside lines 120-124>", "new_text": "<replacement lines>"}
- Lines start..end (1-based, inclusive) are replaced by new_text. end = start - 1 inserts before start.
  new_text "" deletes the lines. Omit new_text to just print the numbered lines.
- expect (recommended) = the old lines being replaced. Matching ignores whitespace/indentation and one level of
  extra escaping. If expect is not inside start..end but is the whole old block, the block is replaced where it
  really is (nearest occurrence); a shorter expect moves the range only when new_text's context lines confirm it.
  Otherwise nothing is written and the real lines are shown.
- Keeps the file's newline style (LF/CRLF); if new_text has no indentation but the old lines had, it is re-indented.
- A .py file that no longer compiles is restored automatically (pass "force": "true" to keep it).
Prints the unified diff and the compile result.
"""
import difflib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locate import find_root, parse_args, truthy  # noqa: E402

MAX_DIFF = 2200


def numbered(lines, a, b):
    return "\n".join("%5d| %s" % (i + 1, lines[i].rstrip("\r\n")[:140]) for i in range(max(0, a), min(len(lines), b)))


def indent_of(s):
    return len(s) - len(s.lstrip(" \t"))


def compile_ok(path, text):
    if not path.endswith(".py"):
        return True, ""
    try:
        compile(text, path, "exec", dont_inherit=True)
        return True, ""
    except SyntaxError as e:
        return False, "SyntaxError line %s: %s | %s" % (e.lineno, e.msg, (e.text or "").strip()[:100])
    except Exception as e:  # ValueError (null bytes) etc.
        return False, "%s: %s" % (type(e).__name__, e)


def squash(s):
    return re.sub(r"\s+", " ", s).strip()


_ESC = {"n": "\n", "t": "\t", '"': '"', "'": "'", "\\": "\\"}


def unescape(s):
    """Undo one level of JSON-style escaping (\\n \\t \\" \\' \\\\) in a single pass; other backslashes stay."""
    if not s or "\\" not in s:
        return s
    return re.sub(r"\\([nt\"'\\])", lambda m: _ESC[m.group(1)], s)


def in_range(lines, start, end, exp):
    """expect occurs inside lines start..end (exact, or whitespace-insensitive)."""
    if start < 1 or end < start - 1:
        return False
    hay = "".join(lines[start - 1:end]).replace("\r\n", "\n")
    exp = exp.replace("\r\n", "\n")
    return exp in hay or (bool(squash(exp)) and squash(exp) in squash(hay))


def block_matches(lines, exp):
    """Where expect matches whole lines (whitespace-insensitive, blank lines ignored): [(first, last)] 1-based."""
    want = [squash(l) for l in exp.replace("\r\n", "\n").split("\n") if l.strip()]
    if not want:
        return []
    nb = [(i + 1, squash(l)) for i, l in enumerate(lines) if l.strip()]
    out = []
    k = len(want)
    for j in range(len(nb) - k + 1):
        if nb[j][1] == want[0] and all(nb[j + m][1] == want[m] for m in range(1, k)):
            out.append((nb[j][0], nb[j + k - 1][0]))
    return out


MAX_SHIFT = 60


def resolve_range(lines, start, end, expect, new_text, bad_range):
    """Return (start, end, unescaped, note) for the lines to replace, or None (= do not write).

    1. expect inside start..end (exact or whitespace-insensitive; also after undoing double escaping) -> as given.
    2. Otherwise the line numbers are off. If expect is the whole old block (>= the range length - 1 complete lines),
       replace exactly where that block is: the occurrence nearest to start (unique, or within MAX_SHIFT lines).
    3. A shorter expect (an anchor inside a longer range) moves the range only if new_text confirms the shift
       (its first or last line equals the shifted range's first or last line), within MAX_SHIFT lines.
    """
    variants = [(expect, False)]
    if unescape(expect) != expect:
        variants.append((unescape(expect), True))
    if not bad_range:
        for exp, unesc in variants:
            if in_range(lines, start, end, exp):
                return start, end, unesc, ""
        if end == start - 1:   # insertion: expect is context next to the insertion point, never replaced
            for exp, unesc in variants:
                if in_range(lines, max(1, start - 3), min(len(lines), start + 2), exp):
                    return start, end, unesc, ""
            return None
    length = max(0, end - start + 1)
    for exp, unesc in variants:
        occ = block_matches(lines, exp)
        if not occ:
            continue
        k = len([l for l in exp.split("\n") if l.strip()])
        nl = new_text.replace("\r\n", "\n")
        if unesc:
            nl = unescape(nl)
        if k >= length - 1:
            gap = lambda ab: max(0, ab[0] - end, start - ab[1])   # distance between the block and the given range
            occ.sort(key=gap)
            a, b = occ[0]
            if len(occ) > 1 and (gap(occ[1]) == gap(occ[0]) or gap(occ[0]) > MAX_SHIFT):
                return None
            return a, b, unesc, "expect found at lines %d-%d (not %d-%d): replaced those lines" % (a, b, start, end)
        new_nb = [squash(l) for l in nl.split("\n") if l.strip()]
        if not new_nb:
            continue
        best = None
        for a, b in occ:
            for d in range(b - end, a - start + 1):   # shifts that put the anchor inside the range
                s2, e2 = start + d, end + d
                if abs(d) > MAX_SHIFT or s2 < 1 or e2 > len(lines):
                    continue
                if squash(lines[s2 - 1]) == new_nb[0] or squash(lines[e2 - 1]) == new_nb[-1]:
                    if best is None or abs(d) < abs(best):
                        best = d
        if best is not None:
            return (start + best, end + best, unesc,
                    "line numbers shifted by %+d to lines %d-%d (expect and new_text's context lines match there)"
                    % (best, start + best, end + best))
    return None


def main(argv):
    opts, pos = parse_args(argv, {"force"})
    g = opts.get
    path = g("path") or g("file") or g("filepath") or g("file_path") or (pos[0] if pos else "")
    start = g("start") or g("start_line") or g("from") or (pos[1] if len(pos) > 1 else None)
    end = g("end") or g("end_line") or g("to") or (pos[2] if len(pos) > 2 else None)
    new_text = None
    for key in ("new_text", "new", "text", "content", "replacement", "new_string", "code"):
        if key in opts:
            new_text = opts[key]
            break
    if new_text is None and len(pos) > 3:
        new_text = pos[3]
    expect = None
    for key in ("expect", "old", "old_text", "old_string", "expected"):
        if key in opts:
            expect = opts[key]
            break
    if new_text is not None and expect is None and len(pos) > 4:
        expect = pos[4]
    if not path or start is None:
        print('usage: args {"path": "f.py", "start": 10, "end": 12, "expect": "old text", "new_text": "new lines"}')
        return
    root = find_root(opts)
    rel = path.replace("\\", "/")
    rel = re.sub(r"^/workspace/", "", rel)
    full = rel if os.path.isabs(rel) else os.path.join(root, rel)
    try:
        start = int(str(start).strip())
        end = int(str(end).strip()) if end is not None and str(end).strip() != "" else start
    except ValueError:
        print("start/end must be integers, got start=%r end=%r" % (start, end))
        return

    if not os.path.exists(full):
        if new_text is not None and start <= 1 and end <= 1:
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            body = new_text if new_text.endswith("\n") or new_text == "" else new_text + "\n"
            with open(full, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(body)
            ok, err = compile_ok(full, body)
            print("created %s (%d lines). compile: %s" % (rel, body.count("\n"), "OK" if ok else err))
            return
        print("file not found: %s (root %s)" % (rel, root))
        return
    with open(full, "rb") as fh:
        raw = fh.read()
    try:
        text = raw.decode("utf-8")
        enc = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
        enc = "latin-1"
    crlf = text.count("\r\n") > (text.count("\n") - text.count("\r\n"))
    nl = "\r\n" if crlf else "\n"
    lines = text.splitlines(True)
    n = len(lines)

    if new_text is None:   # viewer mode
        a, b = start - 1, max(end, start)
        print("%s lines %d-%d of %d:\n%s" % (rel, start, min(b, n), n, numbered(lines, a, b)))
        return
    bad_range = start < 1 or start > n + 1 or end > n or end < start - 1
    has_expect = expect is not None and bool(expect.strip())
    if bad_range and not has_expect:
        print("bad range start=%d end=%d for %s with %d lines (end = start-1 inserts)." % (start, end, rel, n))
        return
    notes = []
    if has_expect:
        res = resolve_range(lines, start, end, expect, new_text, bad_range)
        if res is None:
            where = []
            for exp in (expect, unescape(expect)):
                first = next((l.strip() for l in exp.replace("\r\n", "\n").splitlines() if l.strip()), "")
                where = [i + 1 for i, l in enumerate(lines) if first and squash(first) in squash(l)][:6]
                if where:
                    break
            print("NOT WRITTEN: expect text not found in lines %d-%d of %s." % (start, end, rel))
            print("Actual lines:\n" + numbered(lines, max(0, start - 3), min(n, end + 2)))
            if where:
                print("expect's first line occurs at line(s): %s. Retry with the right start/end." % ", ".join(map(str, where)))
            return
        start, end, unesc, note = res
        if note:
            notes.append(note)
        if unesc and unescape(new_text) != new_text:
            new_text = unescape(new_text)
            notes.append("expect and new_text were escaped twice (\\\\n, \\\") - unescaped both")
    old_seg = lines[start - 1:end]

    # normalise new text: newline style, trailing newline, indentation
    new = new_text.replace("\r\n", "\n")
    if new and "\n" not in new and "\\n" in new and end > start:
        cand = new.replace("\\n", "\n")
        trial = "".join(lines[:start - 1]) + cand + "\n" + "".join(lines[end:])
        if compile_ok(full, trial)[0] and not compile_ok(full, "".join(lines[:start - 1]) + new + "\n" + "".join(lines[end:]))[0]:
            new = cand
            notes.append("converted literal \\n sequences to line breaks")
    new_lines = new.split("\n")
    if new_lines and new_lines[-1] == "":
        new_lines = new_lines[:-1]
    old_nonblank = [l for l in old_seg if l.strip()]
    new_nonblank = [l for l in new_lines if l.strip()]
    if end >= n and lines and not lines[-1].endswith(("\n", "\r")) and new_lines:
        tail_nl = ""   # replacing the last line of a file without a trailing newline
    else:
        tail_nl = nl

    def build(nls):
        return "".join(lines[:start - 1]) + nl.join(nls) + (tail_nl if nls else "") + "".join(lines[end:])

    result = build(new_lines)
    if old_nonblank and new_nonblank:
        old_min = min(indent_of(l.rstrip("\r\n")) for l in old_nonblank)
        new_min = min(indent_of(l) for l in new_nonblank)
        if new_min == 0 and old_min > 0:
            # new_text probably lost its indentation; re-indent only if that is what makes the file compile
            pad = next(re.match(r"[ \t]*", l).group(0) for l in old_nonblank if indent_of(l.rstrip("\r\n")) == old_min)
            rest = new_nonblank[1:]
            first_only = bool(rest) and min(indent_of(l) for l in rest) > old_min
            if first_only:
                # only the first line lost its indentation (the rest is nested deeper than the old block)
                i0 = next(i for i, l in enumerate(new_lines) if l.strip())
                alt = list(new_lines)
                alt[i0] = re.match(r"[ \t]*", old_nonblank[0]).group(0) + alt[i0]
                why = "re-indented the first new line like the old first line"
            else:
                alt = [(pad + l) if l.strip() else l for l in new_lines]
                why = "re-indented new_text by %d columns to match the old lines" % old_min
            alt_result = build(alt)
            if full.endswith(".py") and (first_only or not compile_ok(full, result)[0]) and compile_ok(full, alt_result)[0]:
                new_lines, result = alt, alt_result
                notes.append(why)
            else:
                notes.append("warning: new_text starts at column 0 but the old lines were indented by %d" % old_min)
    if old_nonblank and old_nonblank[0].startswith("\t") and new_nonblank and not any(l.startswith("\t") for l in new_lines):
        notes.append("warning: old lines are tab-indented, new_text uses spaces")
    ok, err = compile_ok(full, result)
    force = truthy(opts.get("force", ""))
    if not ok and not force and compile_ok(full, text)[0]:
        print("NOT WRITTEN: the edit would break the file (%s). File unchanged." % err)
        m = re.search(r"line (\d+)", err)
        if m:
            ln = int(m.group(1))
            rl = result.splitlines(True)
            print("Result around that line:\n" + numbered(rl, ln - 4, ln + 3))
        return
    with open(full, "wb") as fh:
        fh.write(result.encode(enc))
    diff = difflib.unified_diff(text.replace("\r\n", "\n").splitlines(True), result.replace("\r\n", "\n").splitlines(True),
                                "a/" + rel, "b/" + rel, n=2)
    d = "".join(diff)
    if len(d) > MAX_DIFF:
        d = d[:MAX_DIFF] + "\n...[diff truncated]\n"
    delta = len(new_lines) - len(old_seg)
    print("WROTE %s: lines %d-%d replaced by %d line(s); later line numbers shift by %+d.%s" % (
        rel, start, end, len(new_lines), delta, (" " + "; ".join(notes)) if notes else ""))
    print(d.rstrip("\n") or "(no textual change)")
    print("compile: %s" % ("OK" if ok else "FAILED (kept because force): " + err))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    try:
        main(sys.argv[1:])
    except Exception as e:
        print("edit_lines.py internal error: %s: %s" % (type(e).__name__, str(e)[:300]))
