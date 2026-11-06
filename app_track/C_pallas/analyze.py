"""Offline analysis (no jax): kill-rate tables, structural graph features, static localisation baselines.
usage: python analyze.py <factory_out_dir> <seeds_dir> <results_dir>
"""
import sys, os, json, ast, re, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutate

out_dir, seeds_dir, res_dir = sys.argv[1:4]
os.makedirs(res_dir, exist_ok=True)
rows = [json.loads(l) for l in open(os.path.join(out_dir, "rows.jsonl"))]
muts = [r for r in rows if r.get("kind") == "mutant"]
seed_fail = [r for r in rows if r.get("kind") != "mutant"]


def base_src(seed):
    p = os.path.join(seeds_dir, seed + ".py")
    if os.path.exists(p):
        return open(p).read()
    return open(os.path.join(out_dir, "lib_src", seed + ".py")).read()


def verified(r):
    return r["status"] == "numerics_fail" and r["status2"] == "numerics_fail"


# ---------------- tables ----------------
def table(keyf):
    agg = collections.defaultdict(collections.Counter)
    for r in muts:
        k = keyf(r)
        agg[k]["n"] += 1
        agg[k][r["status"]] += 1
        agg[k]["flaky"] += int(r["flaky"])
        agg[k]["verified"] += int(verified(r))
        agg[k]["sec"] += r["sec"] / 2
    return agg


lines = []
byop = table(lambda r: r["site"]["op"])
lines.append("| operator | class | candidates | verified bug | survived (ok) | crash | shape-mismatch | timeout | flaky | kill rate (verified/cand) | mean s/run |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
for op in mutate.OPS:
    a = byop.get(op)
    if not a:
        lines.append("| %s | %s | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | - |" % (op, mutate.CLASS[op]))
        continue
    lines.append("| %s | %s | %d | %d | %d | %d | %d | %d | %d | %.2f | %.1f |" % (
        op, mutate.CLASS[op], a["n"], a["verified"], a["ok"], a["crash"], a["shape"], a["timeout"], a["flaky"],
        a["verified"] / a["n"], a["sec"] / a["n"]))
tot = collections.Counter()
for a in byop.values():
    tot.update(a)
lines.append("| **all** | | %d | %d | %d | %d | %d | %d | %d | %.2f | %.1f |" % (
    tot["n"], tot["verified"], tot["ok"], tot["crash"], tot["shape"], tot["timeout"], tot["flaky"], tot["verified"] / tot["n"], tot["sec"] / tot["n"]))
open(os.path.join(res_dir, "table_operators.md"), "w").write("\n".join(lines) + "\n")

byseed = table(lambda r: r["seed"])
l2 = ["| seed | candidates | verified | survived | crash | shape | timeout | flaky |", "|---|---|---|---|---|---|---|---|"]
for s in sorted(byseed):
    a = byseed[s]
    l2.append("| %s | %d | %d | %d | %d | %d | %d | %d |" % (s, a["n"], a["verified"], a["ok"], a["crash"], a["shape"], a["timeout"], a["flaky"]))
open(os.path.join(res_dir, "table_seeds.md"), "w").write("\n".join(l2) + "\n")

byclass = table(lambda r: mutate.CLASS[r["site"]["op"]])
l3 = ["| bug class | candidates | verified |", "|---|---|---|"]
for c, a in sorted(byclass.items()):
    l3.append("| %s | %d | %d |" % (c, a["n"], a["verified"]))
open(os.path.join(res_dir, "table_classes.md"), "w").write("\n".join(l3) + "\n")

# crash exception breakdown
exc = collections.Counter()
for r in muts:
    if r["status"] == "crash":
        exc[(r["site"]["op"], r["diag"].get("exc"), r["diag"].get("msg", "")[:70])] += 1
open(os.path.join(res_dir, "crash_reasons.txt"), "w").write(
    "\n".join("%4d  %-12s %-22s %s" % (c, k[0], k[1], k[2]) for k, c in exc.most_common(40)) + "\n")

V = [r for r in muts if verified(r)]
secs = [r["sec"] / 2 for r in muts]
stats = dict(n_seeds=len(set(r["seed"] for r in muts)), n_seed_fail=len(seed_fail), n_candidates=len(muts),
             n_verified=len(V), n_flaky=sum(r["flaky"] for r in muts), median_sec=float(np.median(secs)),
             p95_sec=float(np.percentile(secs, 95)), total_sec_both_runs=float(sum(r["sec"] for r in muts)),
             verified_nonfinite=sum(1 for r in V if not r["diag"].get("finite", True)),
             verified_by_seed={s: byseed[s]["verified"] for s in byseed},
             seed_failures=[(r["seed"], r.get("status") or r.get("error")) for r in seed_fail])
json.dump(stats, open(os.path.join(res_dir, "stats.json"), "w"), indent=1)

# ---------------- structural graph ----------------
def build_graph(src):
    t = ast.parse(src)
    cx = mutate._Ctx(src)
    nodes, edges = [], []  # nodes (id, type, line, name); edges (src,dst,type)
    def add(n, typ, name=""):
        nodes.append(dict(id=len(nodes), type=typ, line=getattr(n, "lineno", 0), name=name, n=n))
        return len(nodes) - 1
    fdefs = {}
    for n in ast.walk(t):
        if isinstance(n, ast.FunctionDef):
            fdefs[n.name] = add(n, "kernel_fn" if n.name in cx.kernel_names else "fn", n.name)
    for n in ast.walk(t):
        if isinstance(n, ast.FunctionDef):
            for c in ast.walk(n):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in fdefs and c.func.id != n.name:
                    edges.append((fdefs[n.name], fdefs[c.func.id], "calls"))
    pcs = []
    for n in ast.walk(t):
        if mutate._is(n, "pallas_call"):
            pc = add(n, "pallas_call")
            pcs.append(pc)
            f = cx.func_of(n)
            if f and f.name in fdefs:
                edges.append((fdefs[f.name], pc, "contains"))
            if n.args:
                a = n.args[0]
                kn = a.id if isinstance(a, ast.Name) else (a.args[0].id if mutate._is(a, "partial") and a.args and isinstance(a.args[0], ast.Name) else None)
                if kn in fdefs:
                    edges.append((pc, fdefs[kn], "kernel_arg"))
            for sub in ast.walk(n):
                if isinstance(sub, ast.keyword) and sub.arg == "grid":
                    g = add(sub.value, "grid")
                    edges.append((pc, g, "grid"))
                if mutate._is(sub, "BlockSpec"):
                    b = add(sub, "blockspec")
                    edges.append((pc, b, "spec"))
                    for lam in mutate._blockspec_lambdas(sub):
                        l = add(lam, "index_map")
                        edges.append((b, l, "index_map"))
    for n in ast.walk(t):
        if mutate._is(n, "program_id"):
            p = add(n, "program_id")
            f = cx.func_of(n)
            if f and f.name in fdefs:
                edges.append((fdefs[f.name], p, "uses_grid_axis"))
            for pc in pcs:
                edges.append((p, pc, "axis_of"))
        if mutate._is(n, "when"):
            w = add(n, "when_guard")
            f = cx.func_of(n)
            if f and f.name in fdefs:
                edges.append((fdefs[f.name], w, "guards"))
    return nodes, edges


gfeat = []
for s in sorted(set(r["seed"] for r in muts)):
    src = base_src(s)
    nodes, edges = build_graph(src)
    tc = collections.Counter(n["type"] for n in nodes)
    gfeat.append(dict(seed=s, n_nodes=len(nodes), n_edges=len(edges), lines=src.count("\n") + 1, **{"n_" + k: v for k, v in tc.items()}))
with open(os.path.join(res_dir, "graph_features.json"), "w") as f:
    json.dump(gfeat, f, indent=1)
gl = ["| seed | lines | nodes | edges | kernel_fn | fn | pallas_call | blockspec | index_map | grid | program_id | when |", "|" + "---|" * 12]
for g in gfeat:
    gl.append("| %s | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d |" % (
        g["seed"], g["lines"], g["n_nodes"], g["n_edges"], g.get("n_kernel_fn", 0), g.get("n_fn", 0), g.get("n_pallas_call", 0),
        g.get("n_blockspec", 0), g.get("n_index_map", 0), g.get("n_grid", 0), g.get("n_program_id", 0), g.get("n_when_guard", 0)))
open(os.path.join(res_dir, "table_graph.md"), "w").write("\n".join(gl) + "\n")

# ---------------- localisation baselines ----------------
PATTERNS = [(r"index_map|lambda [a-z_, ]+:\s*\(", 2.0), (r"BlockSpec", 1.5), (r"program_id", 1.5), (r"\bgrid\b", 1.5),
            (r"pl\.when|pl\.ds|fori_loop", 1.5), (r"jnp\.where|\bmask", 1.2), (r"axis\s*=|keepdims", 1.0),
            (r"preferred_element_type|astype|bfloat16", 1.0), (r"zeros|ones|full\(", 0.8), (r"\bscale|eps\b", 0.6),
            (r"cdiv|//", 0.8), (r"jnp\.dot|@", 0.5), (r"[<>]=?", 0.4)]


def code_lines(src):
    t = ast.parse(src)
    ls = set()
    for n in ast.walk(t):
        if isinstance(n, (ast.stmt, ast.expr)) and hasattr(n, "lineno"):
            ls.add(n.lineno)
    # decorator lines for pl.when are expr nodes already; drop docstring/def header lines noise is kept
    return sorted(ls)


def prior_scores(src, cl):
    L = src.split("\n")
    return {l: sum(w for p, w in PATTERNS if re.search(p, L[l - 1])) for l in cl}


def rank_of(scores, gold, rng, key=None):
    # rank of gold line with random tie-breaking (expected rank, handled by random jitter)
    items = list(scores.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: -kv[1])
    for i, (l, _) in enumerate(items):
        if l == gold:
            return i + 1
    return None


FEATS = ["frac_bad", "rows_bad", "cols_bad", "first_half_bad", "last_block_only", "finite", "out_zero_frac", "ndim"]


def featvec(d):
    me = d.get("max_err")
    return [d.get(k, 0) if not isinstance(d.get(k, 0), bool) else float(d.get(k)) for k in FEATS] + [np.log10((me or 1e30) + 1e-12)]


classes = sorted(set(mutate.CLASS.values()))
from sklearn.ensemble import RandomForestClassifier
X = np.array([featvec(r["diag"]) for r in V])
Y = np.array([mutate.CLASS[r["site"]["op"]] for r in V])
G = np.array([r["seed"] for r in V])
probs = {}
for s in set(G):
    tr, te = G != s, G == s
    if len(set(Y[tr])) < 2:
        continue
    clf = RandomForestClassifier(n_estimators=200, random_state=0, min_samples_leaf=2).fit(X[tr], Y[tr])
    pr = clf.predict_proba(X[te])
    for idx, p in zip(np.where(te)[0], pr):
        probs[idx] = dict(zip(clf.classes_, p))
cls_acc = float(np.mean([max(probs[i], key=probs[i].get) == Y[i] for i in probs])) if probs else None
maj = collections.Counter(Y).most_common(1)[0][1] / len(Y)

rng = random.Random(0)
res = collections.defaultdict(list)       # method -> list of (rank_line, rank_stmt_func)
cand_stats = []
site_cache = {}
per_op = collections.defaultdict(lambda: collections.defaultdict(list))
for i, r in enumerate(V):
    s = r["site"]
    src = base_src(r["seed"])
    if r["seed"] not in site_cache:
        site_cache[r["seed"]] = (code_lines(src), prior_scores(src, code_lines(src)), mutate.find_sites(src), build_graph(src))
    cl, prior, sites, (gn, ge) = site_cache[r["seed"]]
    gold = s["line"]
    if gold not in prior:
        continue
    cand_stats.append(len(cl))
    # B0 random
    sc0 = {l: 0.0 for l in cl}
    # B1 pallas-token prior
    sc1 = dict(prior)
    # B2 operator-catalogue x diagnostic classifier: each catalogue site contributes class prob
    sc2 = {l: 0.0 for l in cl}
    sc2b = {l: 0.0 for l in cl}  # catalogue only (uniform class prob)
    pr = probs.get(i, {})
    for st in sites:
        l = st["line"]
        if l in sc2:
            sc2[l] = max(sc2[l], pr.get(mutate.CLASS[st["op"]], 0.0))
            sc2b[l] = max(sc2b[l], 1.0)
    # B3 graph propagation: add 0.5 * mean prior of graph-neighbour nodes (typed edges) to nodes' lines
    line_nb = collections.defaultdict(list)
    for a, b, t in ge:
        la, lb = gn[a]["line"], gn[b]["line"]
        line_nb[la].append(lb)
        line_nb[lb].append(la)
    sc3 = {}
    for l in cl:
        nb = line_nb.get(l, [])
        sc3[l] = sc2[l] + 0.05 * (np.mean([sc1.get(x, 0) for x in nb]) if nb else 0.0) + 0.001 * sc1[l]
    # lines of the same statement as gold count as hit for stmt-level; function-level via func name
    for name, sc in (("B0_random", sc0), ("B1_pallas_token_prior", sc1), ("B2a_catalogue_only", sc2b),
                     ("B2b_catalogue+diag_classifier", sc2), ("B3_B2b+graph_propagation", sc3)):
        rk = rank_of(sc, gold, rng)
        res[name].append(rk)
        per_op[name][s["op"]].append(rk)

summary = {}
for name, rk in res.items():
    rk = np.array(rk, dtype=float)
    summary[name] = dict(n=len(rk), acc1=float((rk <= 1).mean()), acc3=float((rk <= 3).mean()), acc5=float((rk <= 5).mean()),
                         acc10=float((rk <= 10).mean()), mrr=float((1 / rk).mean()), median_rank=float(np.median(rk)))
summary["_meta"] = dict(mean_candidate_lines=float(np.mean(cand_stats)), diag_class_acc_leave_one_seed_out=cls_acc,
                        majority_class_rate=float(maj), n_verified=len(V))
json.dump(summary, open(os.path.join(res_dir, "localisation.json"), "w"), indent=1)
ll = ["| method | n | acc@1 | acc@3 | acc@5 | acc@10 | MRR | median rank |", "|---|---|---|---|---|---|---|---|"]
for name, d in summary.items():
    if name.startswith("_"):
        continue
    ll.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.0f |" % (name, d["n"], d["acc1"], d["acc3"], d["acc5"], d["acc10"], d["mrr"], d["median_rank"]))
ll.append("")
ll.append("mean candidate lines per task: %.1f; diagnostic->bug-class accuracy (leave-one-seed-out RF): %s vs majority %.3f" % (
    summary["_meta"]["mean_candidate_lines"], cls_acc, maj))
open(os.path.join(res_dir, "table_localisation.md"), "w").write("\n".join(ll) + "\n")
# per-operator MRR for B2b
lo = ["| operator | n | B1 MRR | B2b MRR |", "|---|---|---|---|"]
for op in mutate.OPS:
    a = per_op["B1_pallas_token_prior"].get(op)
    b = per_op["B2b_catalogue+diag_classifier"].get(op)
    if a:
        lo.append("| %s | %d | %.3f | %.3f |" % (op, len(a), np.mean([1 / x for x in a]), np.mean([1 / x for x in b])))
open(os.path.join(res_dir, "table_loc_by_op.md"), "w").write("\n".join(lo) + "\n")
print(open(os.path.join(res_dir, "table_operators.md")).read())
print(json.dumps(stats, indent=1)[:1500])
print("\n".join(ll))
print("\n".join(lo))
