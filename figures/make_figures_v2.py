"""Three figures for the 3,000-word writeup. Run:  python paper/make_figures_v2.py [traces.tar.gz]
CPU only. Needs numpy + matplotlib. Writes paper/figures/fig_time_anatomy.png, fig_prose_vs_mech.png, fig_noise_power.png
Everything is recomputed from raw files: A/B session-1 traces tarball (or research/top10/anatomy/runs_slim.json when the tar is absent),
datasets/gemma4-lb-eval-ckpt/results_*.jsonl, research/submissions_history.csv, and the simulation outputs in research/top10/strategy/.
Deterministic (bootstrap seed fixed)."""
import csv, importlib.util, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
P = lambda *a: os.path.join(ROOT, *a)
TAR_DEFAULT = ("<home>/AppData/Local/Temp/claude/<proj>-"
               "Gemma-4-developer-agent-paper/ec71eb7d-f89d-4a42-b2fa-abd700099f6a/scratchpad/ab_s1_traces/traces_lbeval-20261007-043225.tar.gz")
TAR = sys.argv[1] if len(sys.argv) > 1 else TAR_DEFAULT
ARMS = ["edit-early-v6-locate", "edit-early-v7-decide-nt", "edit-early-v7-lean-nt"]
LAB = {ARMS[0]: "v6 control", ARMS[1]: "decide-nt\n(prompt rules)", ARMS[2]: "lean-nt\n(budget)"}
# Okabe-Ito (colour-blind safe)
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00", sky="#56B4E9", pink="#CC79A7",
          yellow="#F0E442", grey="#999999", black="#000000")
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e4e4",
                     "axes.axisbelow": True, "legend.frameon": False})
RNG = np.random.default_rng(20261010)


def footer(fig, text):
    fig.text(0.01, 0.005, text, fontsize=7, color="#555", ha="left", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, facecolor="white")
    plt.close(fig)
    print("wrote", name, os.path.getsize(os.path.join(OUT, name)) // 1024, "KB")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def boot_ci(x, B=4000):
    x = np.asarray(x, float)
    s = [x[RNG.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return np.percentile(s, [2.5, 97.5])


# ------------------------------------------------------------------ load runs (per-call numbers)
def load_runs():
    if os.path.exists(TAR):
        spec = importlib.util.spec_from_file_location("br", P("research", "top10", "anatomy", "build_runs.py"))
        br = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(br)
        R = br.build(TAR)
        src = "A/B session-1 trace tarball via research/top10/anatomy/build_runs.py"
    else:
        R = json.load(open(P("research", "top10", "anatomy", "runs_slim.json")))
        src = "research/top10/anatomy/runs_slim.json"
    rows = {}
    for a in ARMS:
        for x in jl(P("datasets", "gemma4-lb-eval-ckpt", f"results_{a}.jsonl")):
            rows[(a, x["instance_id"])] = x
    for r in R:
        r["row"] = rows[(r["arm"], r["task"])]
        r["wall"] = r["row"]["agent_min"] * 60
    return R, rows, src


R, ROWS, SRC1 = load_runs()
assert len(R) == 40, len(R)

# ------------------------------------------------------------------ Figure 1: time anatomy
X, Y = [], []
for r in R:
    r["prim"] = [c for c in r["calls"] if not c["compact"] and c.get("exc") is None]
    for c in r["prim"]:
        X.append([1, c["ct"], c["pt"]])
        Y.append(c["lat"])
X, Y = np.array(X, float), np.array(Y, float)
coef = np.linalg.lstsq(X, Y, rcond=None)[0]
r2_full = 1 - ((Y - X @ coef) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
ct, lat = X[:, 1], Y
b1, a1 = np.polyfit(ct, lat, 1)
r2_ct = np.corrcoef(ct, lat)[0, 1] ** 2
A_, B_, C_ = coef
CATS = [("reasoning decode", OI["blue"]), ("tool-call args decode", OI["sky"]), ("visible content decode", OI["yellow"]),
        ("per-call fixed + prefill", OI["grey"]), ("compaction summariser", OI["pink"]),
        ("in-flight call cut at time cap", OI["verm"]), ("tool execution", OI["green"]), ("idle / harness gap", OI["black"])]
comp = {a: [] for a in ARMS}
for r in R:
    prim = r["prim"]
    rsn = sum(min(c["rsn_tok"], c["ct"]) for c in prim)
    tool_t = cont_t = 0.0
    for c in prim:
        other = max(c["ct"] - c["rsn_tok"], 0)
        ch = c["tc_chars"] + c["content_chars"]
        if ch > 0 and other > 0:
            tool_t += other * c["tc_chars"] / ch
            cont_t += other * c["content_chars"] / ch
    llm_p = sum(c["lat"] for c in prim)
    d_r, d_t, d_c = B_ * rsn, B_ * tool_t, B_ * cont_t
    fixed = llm_p - d_r - d_t - d_c   # per-call intercept + prefill + small unattributed residual
    comp_t = sum(c["lat"] for c in r["calls"] if c["compact"] and c.get("exc") is None)
    killed = sum(c["lat"] for c in r["calls"] if c.get("exc") is not None)
    tool = sum(c["tool_s"] for c in prim if c.get("tool_s") is not None)
    idle = r["wall"] - llm_p - comp_t - killed - tool
    comp[r["arm"]].append([d_r, d_t, d_c, fixed, comp_t, killed, tool, idle, r["wall"]])
means = {a: np.mean(comp[a], axis=0) for a in ARMS}
nrun = {a: len(comp[a]) for a in ARMS}
allm = np.mean(sum((comp[a] for a in ARMS), []), axis=0)
xs = np.arange(3)

fig = plt.figure(figsize=(9.2, 6.6))
ax = fig.add_axes([0.075, 0.38, 0.40, 0.50])
ax2 = fig.add_axes([0.60, 0.38, 0.37, 0.50])
bottom = np.zeros(3)
for k, (name, col) in enumerate(CATS):
    v = np.array([means[a][k] for a in ARMS])
    ax.bar(xs, v, 0.62, bottom=bottom, color=col, label=name, edgecolor="white", linewidth=0.5)
    bottom += v
for i, a in enumerate(ARMS):
    w = np.array(comp[a])[:, 8]
    ci = boot_ci(w)
    ax.errorbar(i, w.mean(), yerr=[[w.mean() - ci[0]], [ci[1] - w.mean()]], color="k", capsize=4, lw=1.3)
    ax.text(i + 0.07, ci[1] + 5, f"{w.mean():.0f} s", ha="left", fontsize=11)
ax.set_xticks(xs)
ax.set_xticklabels([LAB[a] + f"\nn={nrun[a]} runs" for a in ARMS], fontsize=10)
ax.set_ylabel("mean wall-clock per run (s)")
ax.set_ylim(0, 350)
ax.grid(axis="x", visible=False)
ax.set_title("(a) Where a 4-minute run goes", loc="left")
h, l = ax.get_legend_handles_labels()
ax.legend(h[::-1], l[::-1], loc="upper center", bbox_to_anchor=(0.62, -0.27), ncol=2, fontsize=9.5,
          handlelength=1.2, columnspacing=1.0)
ax2.scatter(ct, lat, s=9, alpha=0.35, color=OI["blue"], edgecolors="none")
xx = np.linspace(0, ct.max(), 50)
ax2.plot(xx, a1 + b1 * xx, color=OI["verm"], lw=2)
ax2.set_xlabel("completion tokens per call")
ax2.set_ylabel("call latency (s)")
ax2.set_title("(b) Latency is decode time", loc="left")
ax2.text(0.04, 0.95, f"{1000 * b1:.1f} ms/token, R$^2$={r2_ct:.3f}\n(+ prompt tokens: R$^2$={r2_full:.3f})\nn={len(ct)} calls",
         transform=ax2.transAxes, va="top", fontsize=11)
fig.suptitle("Tool execution is %.1f%% of wall-clock; decoding reasoning tokens is %.0f%%"
             % (100 * allm[6] / allm[8], 100 * allm[0] / allm[8]), x=0.01, y=0.985, ha="left", fontsize=13.5, fontweight="bold")
footer(fig, "Source: " + SRC1 + ";\nwall = agent_min in datasets/gemma4-lb-eval-ckpt/results_*.jsonl. Whiskers: bootstrap 95% CI of mean wall.")
save(fig, "fig_time_anatomy.png")
FIG1 = dict(arms={a: dict(n=nrun[a], **dict(zip([c[0] for c in CATS] + ["wall"], np.round(means[a], 1).tolist()))) for a in ARMS},
            all=dict(zip([c[0] for c in CATS] + ["wall"], np.round(allm, 1).tolist())),
            slope_ms_per_tok=1000 * b1, r2_ct=r2_ct, r2_full=r2_full, n_calls=len(ct))

# ------------------------------------------------------------------ Figure 2: prose vs mechanism
tasks = sorted(set.intersection(*[{t for (a, t) in ROWS if a == arm} for arm in ARMS]))
assert len(tasks) == 13, tasks
pre = {a: [] for a in ARMS}
capped = {a: 0 for a in ARMS}
never = {a: 0 for a in ARMS}
cens = {a: [] for a in ARMS}
for t in tasks:
    for a in ARMS:
        x = ROWS[(a, t)]
        pre[a].append(x["pre_edit_read"] + x["pre_edit_search"] + x["pre_edit_locate"])
        cens[a].append(x["first_edit_idx"] is None)
        capped[a] += (x["call_summary"].get("n_primary_len") or 0)
        never[a] += x["first_edit_idx"] is None
fig = plt.figure(figsize=(9.2, 5.2))
ax = fig.add_axes([0.11, 0.17, 0.46, 0.66])
bx = fig.add_axes([0.70, 0.17, 0.27, 0.66])
M = np.array([pre[a] for a in ARMS])
C = np.array([cens[a] for a in ARMS])
for j in range(len(tasks)):
    ax.plot(xs, M[:, j], color="#bbb", lw=0.9, zorder=1)
for i, a in enumerate(ARMS):
    jit = (np.arange(len(tasks)) - 6) * 0.012
    ok = ~C[i]
    ax.scatter(xs[i] + jit[ok], M[i][ok], s=36, color=OI["blue"], zorder=3, label="edited" if i == 0 else None)
    ax.scatter(xs[i] + jit[~ok], M[i][~ok], s=50, facecolors="white", edgecolors=OI["verm"], linewidths=1.8, zorder=3,
               label="never edited (all calls)" if i == 0 else None)
    m = M[i].mean()
    ci = boot_ci(M[i])
    ax.errorbar(xs[i] + 0.22, m, yerr=[[m - ci[0]], [ci[1] - m]], fmt="D", color="k", capsize=4, ms=7, zorder=4,
                label="mean, bootstrap 95% CI" if i == 0 else None)
    ax.text(xs[i] + 0.30, m, f"{m:.1f}", va="center", fontsize=11)
ax.set_xticks(xs)
ax.set_xticklabels([LAB[a] for a in ARMS])
ax.set_xlim(-0.4, 2.6)
ax.set_ylim(0, 44)
ax.set_ylabel("reads + searches before first edit\n(calls per task)")
ax.legend(loc="upper right", fontsize=9.5)
ax.set_title("(a) Per-task, paired, n=13 tasks", loc="left")
ax.grid(axis="x", visible=False)
w = 0.36
bx.bar(xs - w / 2, [capped[a] for a in ARMS], w, color=OI["orange"], label="calls hitting\nmax_output")
bx.bar(xs + w / 2, [never[a] for a in ARMS], w, color=OI["verm"], label="runs never\nediting")
for i, a in enumerate(ARMS):
    bx.text(i - w / 2, capped[a] + 0.12, str(capped[a]), ha="center")
    bx.text(i + w / 2, never[a] + 0.12, str(never[a]), ha="center")
bx.set_xticks(xs)
bx.set_xticklabels(["v6", "decide", "lean"])
bx.set_ylabel("count over 13 runs")
bx.set_ylim(0, 6.5)
bx.legend(loc="upper right", fontsize=9.5)
bx.set_title("(b) Failure counts", loc="left")
bx.grid(axis="x", visible=False)
fig.suptitle("Pre-edit exploration barely changes across arms (13 triples, one sample each)",
             x=0.01, y=0.985, ha="left", fontsize=13, fontweight="bold")
footer(fig, "Source: datasets/gemma4-lb-eval-ckpt/results_edit-early-{v6-locate,v7-decide-nt,v7-lean-nt}.jsonl\n"
            "(pre_edit_read+search+locate, first_edit_idx, call_summary.n_primary_len); locate.py counted as a search.")
save(fig, "fig_prose_vs_mech.png")
FIG2 = dict(tasks=tasks, mean_pre={a: float(np.mean(pre[a])) for a in ARMS},
            median_pre={a: float(np.median(pre[a])) for a in ARMS}, capped=capped, never=never,
            pre_values={a: pre[a] for a in ARMS})

# ------------------------------------------------------------------ Figure 3: noise / power
S3 = json.load(open(P("research", "top10", "strategy", "s3_hit.json")))
S2 = json.load(open(P("research", "top10", "strategy", "s2_posterior.json")))["six_plus_zip_replicates"]
svals = sorted({float(k.split("s=")[1]) for k in S3})
Ns = [1, 10, 20, 55]
cols = {1: OI["black"], 10: OI["blue"], 20: OI["orange"], 55: OI["verm"]}
hist = list(csv.DictReader(open(P("research", "submissions_history.csv"), encoding="utf-8")))


def k_of(score):   # public scores are displayed truncated to 2 dp: k/58 -> floor(100 k/58)/100
    return [k for k in range(59) if math.floor(100 * k / 58 + 1e-9) / 100 == round(score, 2)]


obs = [(h["publicScore"], k_of(float(h["publicScore"]))) for h in hist if h["publicScore"].strip()]
assert all(len(k) == 1 for _, k in obs), obs
obs_k = [k[0] for _, k in obs]
zip_k = [7, 6, 4]
fig = plt.figure(figsize=(9.2, 5.5))
ax = fig.add_axes([0.10, 0.30, 0.87, 0.58])
lo, hi = S2["mu0_ci"]
ax.axvspan(lo / 58, hi / 58, color=OI["sky"], alpha=0.25, lw=0)
ax.axvline(S2["mu0_mean"] / 58, color=OI["sky"], lw=1.5)
for model, ls in (("A", "-"), ("B", "--")):
    for N in Ns:
        y = [S3[f"{model}|same|s={s}"][f"P12_N{N}"] for s in svals]
        ax.plot(svals, y, ls, color=cols[N], lw=2, marker="o" if model == "A" else None, ms=3.5)
ax.axhline(0.5, color="#bbb", lw=0.8)
ax.set_xlim(0.045, 0.21)
ax.set_ylim(0, 1.04)
ax.set_xlabel("true score of the configuration (fraction of 58 public tasks)", labelpad=48)
ax.set_ylabel("P(best of N identical resubmissions\nscores at least 12/58)")
yr = -0.17
ax.scatter([k / 58 for k in obs_k], [yr] * len(obs_k), marker="|", s=380, color=OI["black"], linewidths=2, clip_on=False, zorder=5)
ax.scatter([k / 58 for k in zip_k], [yr - 0.1] * 3, marker="v", s=55, color=OI["verm"], clip_on=False, zorder=5)
ax.text(0.2045, yr, "our %d scored submissions (k/58)" % len(obs_k), ha="right", va="center", fontsize=9.5, clip_on=False)
ax.text(0.2045, yr - 0.1, "identical ZIP, 3 replicates (7, 6, 4 of 58)", ha="right", va="center", fontsize=9.5, color=OI["verm"], clip_on=False)
ax.text((lo + hi) / 2 / 58, 0.30, "posterior of\nour true score\n(%.1f to %.1f\nof 58 tasks)" % (lo, hi), fontsize=10.5, color="#1b6ea8", ha="center", va="center")
hl = [Line2D([0], [0], color=cols[N], lw=2.2, label=f"N={N}") for N in Ns] + \
     [Line2D([0], [0], color="#444", lw=2, ls="-", marker="o", ms=3.5, label="model A"),
      Line2D([0], [0], color="#444", lw=2, ls="--", label="model B")]
ax.legend(handles=hl, loc="upper left", ncol=2, fontsize=10.5)
fig.suptitle("A 12/58 from resubmitting the same file is a lottery ticket, not evidence", x=0.01, y=0.985, ha="left",
             fontsize=13.5, fontweight="bold")
footer(fig, "Source: research/top10/strategy/s3_hit.json (1,500 simulated sets per point, 'same' mode; A: 20 live tasks, kappa 0.5; "
            "B: 30 live tasks, kappa 1.0),\ns2_posterior.json (95% interval of common mean), research/submissions_history.csv.")
save(fig, "fig_noise_power.png")
FIG3 = dict(obs_k=sorted(obs_k), zip_k=zip_k, post_mean_tasks=S2["mu0_mean"], post_ci_tasks=[lo, hi],
            P12={f"{m}|N={N}|s=0.14": S3[f"{m}|same|s=0.14"][f"P12_N{N}"] for m in "AB" for N in Ns})
print(json.dumps(dict(fig1=FIG1, fig2=FIG2, fig3=FIG3), indent=1))
