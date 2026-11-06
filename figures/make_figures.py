"""Generate every figure and derived statistic used in paper/draft.md.

Run from the repo root:   python paper/make_figures.py
Needs: matplotlib, numpy. CPU only, reads small files committed to the repo, writes only to paper/figures/.

Every number plotted is read from a repo file (path noted per figure). The only literals are
(a) results quoted from research/starlette-compat/FINDINGS.md (that study has no machine-readable output in
the repo), and (b) public LB scores from submissions_history.csv. Nothing is fitted or invented.
"""
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)


def P(*parts):
    return os.path.join(ROOT, *parts)


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- style (reference palette, light mode)
C1, C2, C3, C4, C5, C7 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"
GREY, INK, INK2 = "#b9b8b2", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


STATS = {}

# ================================================================ Fig 1: gold/empty-patch control by repo
# source: research/gold_control/gold_control_summary.json, gold_control_tasks.jsonl
gc = json.load(open(P("research", "gold_control", "gold_control_summary.json"), encoding="utf-8"))
tasks = jl(P("research", "gold_control", "gold_control_tasks.jsonl"))
GROUP = {
    "ModuleNotFoundError:typing_inspection": "missing wheel (typing_inspection)",
    "ModuleNotFoundError:dirty_equals": "missing test dep (dirty_equals / attr)",
    "ModuleNotFoundError:attr": "missing test dep (dirty_equals / attr)",
    "requests_src_layout_shadowed_by_installed_requests_wheel": "installed wheel shadows src/",
    "rich_pygments_highlight_drift": "other (pygments drift, SSL, version gate, collection)",
    "ssl_missing_authority_key_identifier": "other (pygments drift, SSL, version gate, collection)",
    "required_test_skipped_or_missing(version_gate?)": "other (pygments drift, SSL, version gate, collection)",
    "collection_error(other)": "other (pygments drift, SSL, version gate, collection)",
}
cats = ["solvable", "passes with empty patch", "missing wheel (typing_inspection)",
        "missing test dep (dirty_equals / attr)", "installed wheel shadows src/",
        "other (pygments drift, SSL, version gate, collection)"]
colors = [C3, C4, C2, C5, C7, GREY]
repos = ["fastapi", "rich", "requests", "httpx"]
counts = {r: Counter() for r in repos}
for t in tasks:
    if t["solvable"]:
        c = "solvable"
    elif t["empty_resolved"]:
        c = "passes with empty patch"
    else:
        c = GROUP.get(t["gold_reason"], "other (pygments drift, SSL, version gate, collection)")
    counts[t["repo"]][c] += 1
assert sum(counts[r]["solvable"] for r in repos) == gc["n_solvable"] == 40
STATS["gold_control"] = {r: dict(counts[r]) for r in repos}

fig, ax = plt.subplots(figsize=(7.0, 2.6))
left = {r: 0 for r in repos}
for c, col in zip(cats, colors):
    vals = [counts[r][c] for r in repos]
    ax.barh(repos, vals, left=[left[r] for r in repos], color=col, edgecolor="white", linewidth=1.5, label=c,
            height=0.6)
    for r, v in zip(repos, vals):
        if v >= 3:
            ax.text(left[r] + v / 2, r, str(v), ha="center", va="center", fontsize=8,
                    color="white" if col not in (C4, GREY, C5) else INK)
        left[r] += v
for r in repos:
    ax.text(left[r] + 0.8, r, f"{counts[r]['solvable']}/{sum(counts[r].values())} solvable", va="center",
            fontsize=8, color=INK2)
ax.invert_yaxis()
ax.set_xlabel("dev tasks")
ax.set_xlim(0, 82)
ax.grid(axis="y", visible=False)
ax.set_title("Gold- and empty-patch control in an emulation of the grader image (129 dev tasks; 40 solvable)",
             loc="left")
ax.legend(ncol=2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32))
save(fig, "fig1_gold_control.png")

# ================================================================ Fig 2: environment changes the fastapi ceiling
# sources: gold_control_summary.json (faithful), research/starlette-compat/FINDINGS.md (leaky, a/b/c headline)
fa = [("gold patch,\nfaithful grader emulation", gc["by_repo"]["fastapi"]["gold_resolved"], C1),
      ("gold patch,\nleaky sandbox (host site-pkgs)", 10, C2),          # FINDINGS.md: condition a = 10/67
      ("gold + starlette-1.x compat,\nleaky sandbox", 31, C2),          # FINDINGS.md: condition b = 31/67
      ("compat edit only (no gold),\nleaky sandbox", 2, GREY)]          # FINDINGS.md: condition c = 2/67
fig, ax = plt.subplots(figsize=(6.2, 2.5))
ax.barh([f[0] for f in fa], [f[1] for f in fa], color=[f[2] for f in fa], height=0.55, edgecolor="white")
for i, f in enumerate(fa):
    ax.text(f[1] + 0.6, i, f"{f[1]}/67", va="center", fontsize=8, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 67)
ax.set_xlabel("fastapi dev tasks resolved by the official verifier")
ax.grid(axis="y", visible=False)
ax.set_title("Same 67 fastapi tasks, same verifier: the sandbox decides the ceiling", loc="left")
save(fig, "fig2_env_sensitivity.png")

# ================================================================ Fig 3: public LB vs calibrated local proxy
# sources: submissions_history.csv ; research/lb-eval/lb_eval_summary.json
rows = list(csv.DictReader(open(P("submissions_history.csv"), encoding="utf-8")))
rows = [r for r in rows if r["publicScore"]]
rows.sort(key=lambda r: r["date"])
lb = json.load(open(P("research", "lb-eval", "lb_eval_summary.json"), encoding="utf-8"))["arms"]["triage-first"]
N_PUB = 58  # public-split size: training/README.md ("LB gap (about 5/58)") [CHECK]
k_s, n_s = lb["resolved_solvable"], 40
k_f, n_f = int(lb["resolved_free"].split("/")[0]), 3
proj = (k_s + k_f) / 129
# SE of the projection: binomial on the solvable stratum (n=40) + FREE stratum (n=3), dev-mix weights fixed
se_proj = math.sqrt((40 / 129) ** 2 * (k_s / n_s) * (1 - k_s / n_s) / n_s
                    + (3 / 129) ** 2 * (k_f / n_f) * (1 - k_f / n_f) / n_f)
STATS["lb_eval"] = {"projected_lb": round(proj, 4), "se_projection": round(se_proj, 4),
                    "solvable_wilson95": [round(x, 3) for x in wilson(k_s, n_s)]}
names = []
fig, ax = plt.subplots(figsize=(7.0, 2.9))
for i, r in enumerate(rows):
    s = float(r["publicScore"])
    se = math.sqrt(s * (1 - s) / N_PUB)
    label = r["description"].split(":")[0].split(" (")[0]
    names.append(label)
    is_tf = r["ref"] == "56823681"
    ax.errorbar(i, s, yerr=1.96 * se, fmt="o", color=C1 if not is_tf else C2, ms=6, capsize=3, lw=1.2)
    ax.text(i - 0.12, s, f"{s:.2f}", fontsize=7, color=INK2, ha="right", va="center")
    if is_tf:
        tf_i = i
STATS["public_lb_binomial_se_at_0.08_n58"] = round(math.sqrt(0.08 * 0.92 / N_PUB), 4)
ax.errorbar(tf_i + 0.35, proj, yerr=1.96 * se_proj, fmt="D", color=C3, ms=6, capsize=3, lw=1.2)
ax.text(tf_i + 0.5, proj, f"lb-eval proxy\n{proj:.3f}", fontsize=7, color=INK, va="center")
ax.set_xticks(range(len(rows)))
ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
ax.set_ylabel("score")
ax.set_ylim(-0.02, 0.2)
ax.set_xlim(-0.6, len(rows) + 0.4)
ax.set_title("Public LB per submission (±1.96·binomial SE, n=58) and the faithful local proxy for triage-first",
             loc="left", fontsize=9)
save(fig, "fig3_lb_calibration.png")

# ================================================================ Fig 4: does the model obey triage? (lb-eval dead tasks)
# source: research/lb-eval/lb_eval_tasks.jsonl
lbt = jl(P("research", "lb-eval", "lb_eval_tasks.jsonl"))
dead = [r for r in lbt if r["category"] in ("env_dead_visible", "dead_other")]
dead.sort(key=lambda r: r["agent_min"])
STATS["triage_obedience"] = {
    "n_dead": len(dead),
    "env_broken_marker": sum(bool(r["env_broken_marker"]) for r in dead),
    "early_exit": sum(bool(r["early_exit"]) for r in dead),
    "fastapi_dead": sum(r["instance_id"].startswith("fastapi") for r in dead),
    "fastapi_marker": sum(bool(r["env_broken_marker"]) for r in dead if r["instance_id"].startswith("fastapi")),
    "solvable_early_exit": sum(bool(r["early_exit"]) for r in lbt if r["category"] == "solvable"),
}
fig, ax = plt.subplots(figsize=(6.6, 2.9))
for i, r in enumerate(dead):
    col = C3 if r["early_exit"] else (C2 if r["env_broken_marker"] else GREY)
    ax.barh(i, r["agent_min"], color=col, height=0.6, edgecolor="white")
    ax.text(r["agent_min"] + 0.06, i, f"{r['tool_calls']} calls", va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(dead)))
ax.set_yticklabels([f"{r['instance_id']}" for r in dead], fontsize=7)
ax.axvline(6.0, color=INK2, lw=1, ls="--")
ax.text(6.02, len(dead) - 0.6, "cap 6.0", fontsize=7, color=INK2)
ax.axvline(1.5, color=INK2, lw=1, ls=":")
ax.text(1.52, len(dead) - 0.6, "design e=1.5", fontsize=7, color=INK2)
ax.set_xlim(0, 7.2)
ax.set_xlabel("agent minutes on a grader-dead task")
ax.grid(axis="y", visible=False)
from matplotlib.patches import Patch  # noqa: E402

ax.legend(handles=[Patch(color=C3, label="early exit (empty, <3 min)"),
                   Patch(color=C2, label="wrote ENV-BROKEN, no early exit"),
                   Patch(color=GREY, label="no ENV-BROKEN marker")], fontsize=7, loc="upper center",
          bbox_to_anchor=(0.45, -0.2), ncol=3)
ax.set_title("triage-first on the 10 dead tasks of the faithful run", loc="left")
save(fig, "fig4_triage_obedience.png")

# ================================================================ Fig 5: schedule risk vs per-task cap
# source: research/schedule-sim/results/schedule_sim.json (rerun on lb-eval timings)
ss = json.load(open(P("research", "schedule-sim", "results", "schedule_sim.json"), encoding="utf-8"))
scen = [("base", "base: dev mix, triage fires 80-100%", C1),
        ("base_halffire", "triage fires 40-70%", C3),
        ("triage_fails", "triage never fires", C4),
        ("triage_fails_slow", "never fires + 30% slower", C2),
        ("ti_fixed_slow", "typing_inspection fixed + 30% slower", C7)]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
for key, lab, col in scen:
    res = ss["results"][key]
    caps = [x["cap"] for x in res]
    axes[0].plot(caps, [x["p_gt_12h"] for x in res], color=col, lw=2, label=lab, marker="o", ms=3)
    axes[1].plot(caps, [x["mean_h"] for x in res], color=col, lw=2, marker="o", ms=3)
axes[0].set_xlabel("per-task cap C (min)")
axes[0].set_ylabel("P(total > 12 h)")
axes[0].set_title("Overrun probability (20k sims)", loc="left")
axes[1].axhline(12, color=INK2, lw=1, ls="--")
axes[1].text(4.8, 12.1, "12 h limit", fontsize=7, color=INK2)
axes[1].set_xlabel("per-task cap C (min)")
axes[1].set_ylabel("mean total hours (N=120)")
axes[1].set_title("Expected wall time", loc="left")
for a in axes:
    a.axvline(6.0, color=GREY, lw=1, ls=":")
    a.axvline(5.5, color=GREY, lw=1, ls=":")
axes[0].legend(fontsize=6.5, loc="upper left")
save(fig, "fig5_schedule_risk.png")

# ================================================================ Fig 6: time-to-resolve survival curves
pools = ss["model"]["S_t_by_pool"]
fig, ax = plt.subplots(figsize=(4.8, 2.7))
for key, lab, col in [("base_pool(ladder+paper15+lbeval)", "pooled (n=205)", C1),
                      ("lbeval_only", "faithful run only (n=51)", C2),
                      ("ladder_only", "ladder run only (n=12)", C3)]:
    d = pools[key]["P_still_running"]
    ts = sorted(d, key=float)
    ax.step([float(t) for t in ts], [d[t] for t in ts], where="post", color=col, lw=2,
            label=lab)
ax.set_xlabel("agent minutes (ladder pace)")
ax.set_ylabel("P(live task still running)")
ax.set_ylim(0, 1.02)
ax.legend(fontsize=7)
ax.set_title("Kaplan-Meier: live tasks not yet finished", loc="left")
save(fig, "fig6_survival.png")

# ================================================================ Fig 7: failure taxonomy (solvable attempts)
# source: research/failure-taxonomy/per_attempt.jsonl
pa = jl(P("research", "failure-taxonomy", "per_attempt.jsonl"))
order = [("resolved", C3), ("no_patch_budget", C2), ("incomplete_fix", C4), ("wrong_location", C7),
         ("right_location_wrong_fix", C5), ("no_patch_gave_up", GREY), ("only_tests_or_scratch", GREY)]
cnt = Counter(r["category"] for r in pa)
STATS["taxonomy_v1"] = dict(cnt)
fig, ax = plt.subplots(figsize=(5.6, 2.6))
labs = [o[0].replace("_", " ") for o in order]
vals = [cnt[o[0]] for o in order]
ax.barh(labs, vals, color=[o[1] for o in order], height=0.6, edgecolor="white")
for i, v in enumerate(vals):
    ax.text(v + 0.2, i, f"{v} ({v / len(pa):.0%})", va="center", fontsize=7, color=INK2)
ax.invert_yaxis()
ax.set_xlabel(f"attempts (n={len(pa)}, 15 distinct rich tasks, 14 arms, leaky sandbox)")
ax.grid(axis="y", visible=False)
ax.set_title("Outcome of attempts on grader-solvable tasks (taxonomy v1)", loc="left")
save(fig, "fig7_failure_taxonomy.png")

# ================================================================ Fig 8: code-graph arms (paired)
# source: datasets/gemma4-paper-ckpt/results_*.jsonl ; research/gold_control/solvable_ids.json
D = {}
for f in glob.glob(P("datasets", "gemma4-paper-ckpt", "results_*.jsonl")):
    rs = jl(f)
    D[rs[0]["arm"]] = rs
sol = set(json.load(open(P("research", "gold_control", "solvable_ids.json"), encoding="utf-8")))
smoke_tasks = sorted(r["instance_id"] for r in D["OFF"])
loc_tasks = sorted(r["instance_id"] for r in D["NOG"])
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [6, 4]})
g1 = ["OFF", "REP", "NOG", "PIPE-BM25", "PIPE-DIR", "PIPE-UNDIR"]
v1 = [sum(r["resolved"] for r in D[a] if r["instance_id"] in smoke_tasks) for a in g1]
axes[0].bar(g1, v1, color=[C1, C1, GREY, C4, C4, C4], edgecolor="white", width=0.6)
for i, v in enumerate(v1):
    axes[0].text(i, v + 0.08, str(v), ha="center", fontsize=8)
axes[0].set_ylim(0, 6)
axes[0].set_ylabel("resolved (leaky sandbox)")
axes[0].set_title("6 fastapi tasks, 15-min cap (all 6 grader-dead)", loc="left", fontsize=8.5)
axes[0].tick_params(axis="x", labelsize=7)
g2 = ["NOG", "LOC-BM25", "LOC-DIR", "LOC-UNDIR"]
v2 = [sum(r["resolved"] for r in D[a]) for a in g2]
v2s = [sum(r["resolved"] for r in D[a] if r["instance_id"] in sol) for a in g2]
axes[1].bar(g2, v2, color=[GREY, C4, C4, C4], edgecolor="white", width=0.6, label="all 30 tasks")
axes[1].bar(g2, v2s, color=C3, edgecolor="white", width=0.6, label="of which grader-solvable (4 tasks)")
for i, (v, vs) in enumerate(zip(v2, v2s)):
    axes[1].text(i, v + 0.2, str(v), ha="center", fontsize=8)
axes[1].set_ylim(0, 30)
axes[1].set_title("30 tasks, 15-min cap", loc="left", fontsize=8.5)
axes[1].tick_params(axis="x", labelsize=7)
axes[1].legend(fontsize=6.5, loc="upper right")
for a in axes:
    a.grid(axis="x", visible=False)
save(fig, "fig8_code_graph_arms.png")

paired = {}
for a in g2[1:]:
    m0 = {r["instance_id"]: r["resolved"] for r in D["NOG"]}
    m1 = {r["instance_id"]: r["resolved"] for r in D[a]}
    paired[a] = {"wins_vs_NOG": sum(m1[t] and not m0[t] for t in loc_tasks),
                 "losses_vs_NOG": sum(m0[t] and not m1[t] for t in loc_tasks)}
STATS["code_graph"] = {"smoke": dict(zip(g1, v1)), "loc_all": dict(zip(g2, v2)),
                       "loc_on_solvable": dict(zip(g2, v2s)), "paired": paired,
                       "n_solvable_in_loc_set": sum(t in sol for t in loc_tasks),
                       "n_solvable_in_smoke_set": sum(t in sol for t in smoke_tasks)}

# ================================================================ Fig 9: official vs repaired graph
# source: kernels/gemma4-build-rep-graphs/output/build_stats.json
bs = json.load(open(P("kernels", "gemma4-build-rep-graphs", "output", "build_stats.json"), encoding="utf-8"))
by = defaultdict(list)
for r in bs:
    by[r["key"].split("_")[0]].append(r)
rl = ["fastapi", "rich", "requests", "httpx"]
mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
off = [mean([x["official_nodes"] for x in by[r]]) for r in rl]
rep = [mean([x["nodes"] for x in by[r]]) for r in rl]
asy = [mean([x["async_nodes"] for x in by[r]]) for r in rl]
rec = [mean([x["official_edge_recall"] for x in by[r]]) for r in rl]
STATS["graph_audit"] = {r: {"n_snapshots": len(by[r]), "official_nodes": round(o), "rep_nodes": round(p),
                            "async_nodes": round(a), "official_edge_recall": round(e, 3)}
                        for r, o, p, a, e in zip(rl, off, rep, asy, rec)}
fig, ax = plt.subplots(figsize=(5.4, 2.6))
x = range(len(rl))
ax.bar([i - 0.18 for i in x], off, width=0.34, color=GREY, label="official graph nodes", edgecolor="white")
ax.bar([i + 0.18 for i in x], rep, width=0.34, color=C1, label="repaired graph nodes", edgecolor="white")
ax.bar([i + 0.18 for i in x], asy, width=0.34, color=C2, label="of which async defs (absent in official)",
       edgecolor="white")
for i in x:
    ax.text(i, rep[i] + 120, f"edge recall {rec[i]:.2f}", ha="center", fontsize=6.5, color=INK2)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{r}\n(n={len(by[r])})" for r in rl], fontsize=7)
ax.set_ylabel("mean nodes per snapshot")
ax.legend(fontsize=6.5)
ax.grid(axis="x", visible=False)
ax.set_title("Competition code graph vs repaired AST graph (127 snapshots)", loc="left")
save(fig, "fig9_graph_audit.png")

with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(STATS, f, indent=1)
print("wrote stats.json")
