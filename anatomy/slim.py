"""runs.json -> runs_slim.json: per-call numbers only (no thought text, no tool output), small enough to commit."""
import json, sys
R = json.load(open(sys.argv[1]))
out = []
for r in R:
    calls = []
    for c in r["calls"]:
        t = (c.get("tool_calls") or [{}])[0] if c.get("tool_calls") else {}
        calls.append({k: c.get(k) for k in ("i", "t_send", "t_end", "lat", "ct", "pt", "rsn_tok", "tc_chars", "content_chars", "finish",
                                           "exc", "compact", "tool_s", "err_kind", "fail2", "nudged_before", "copy_obs", "n_shingles")}
                     | {"tool": t.get("name"), "sub": t.get("sub"), "kind": t.get("kind")})
    out.append({"arm": r["arm"], "task": r["task"], "calls": calls})
json.dump(out, open(sys.argv[2], "w"), separators=(",", ":"))
