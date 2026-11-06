"""one-off: build locate_body.py = v6 locate.py + _body_block.py + render changes"""
import ast
from pathlib import Path

H = Path(__file__).parent
V6 = H.parents[2] / "agent_arms/edit-early-v6-locate/skills/swe-tools/scripts/locate.py"
s = V6.read_text(encoding="utf-8")
blk = (H / "_body_block.py").read_text(encoding="utf-8")
assert "\n\ndef render(R, top, final, terms, phrase_hits):" in s
s = s.replace("\n\ndef render(R, top, final, terms, phrase_hits):", blk + "\n\ndef render(R, top, final, terms, phrase_hits):", 1)
old = '''    out.append("Next: read the best hit (sed -n 'A,Bp' file), state PLAN, edit it.")
    s = "\\n".join(out)
    if len(s) > MAX_OUT:
        s = s[:MAX_OUT] + "\\n...[truncated]"
    return s
'''
assert old in s
new = '''    if top and body_on():
        return with_body(R, top, terms, phrase_hits, out, final)
    out.append("Next: read the best hit (sed -n 'A,Bp' file), state PLAN, edit it.")
    s = "\\n".join(out)
    if len(s) > MAX_OUT:
        s = s[:MAX_OUT] + "\\n...[truncated]"
    return s


LAST_PLAN = []
NEXT_BODY = ("Next: the code of the best hit is printed above with line numbers. If the buggy line is in it, state PLAN and edit "
             "it with those line numbers; otherwise sed -n 'A,Bp' file for another hit.")


def with_body(R, top, terms, phrase_hits, out, final=None):
    """the v6 output (cut at MAX_OUT) + the numbered code of the best hit(s) + the Next line; whole output <= BODY['total']"""
    main = "\\n".join(out)
    if len(main) > MAX_OUT:
        main = main[:MAX_OUT] + "\\n...[truncated]"
    n = int(BODY["n"])
    while True:
        plan = body_plan(R, top, terms, phrase_hits, n, final)
        s = main + "\\n" + "\\n".join(render_body(R, plan)) + "\\n" + NEXT_BODY
        if len(s) <= BODY["total"] or n <= 10:
            break
        n -= 5
    if len(s) > BODY["total"]:
        s = s[:BODY["total"]] + "\\n...[truncated]"
    global LAST_PLAN
    LAST_PLAN = plan      # (read by the benchmark)
    return s
'''
s = s.replace(old, new, 1)
ast.parse(s)
(H / "locate_body.py").write_text(s, encoding="utf-8", newline="\n")
print("ok", len(s))
