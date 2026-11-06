#!/usr/bin/env python3
"""Pre-registered analysis for Candidate A (PREREG.md).  python analyze.py <features_tasks.jsonl> <outdir>

Reads the per-task table written by kernels/gemma4-app-a-features (numbers only) and
research/teacher-gen/C-behaviour/per_run_table.jsonl (agent outcomes).  Writes results.json and tables.md.
"""
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEED = 20261010
NBOOT = 2000
CGRID = [0.03, 0.1, 0.3, 1, 3]
OPS = ["cmp_flip", "boolop_flip", "binop_flip", "augassign_flip", "const_off_by_one", "bound_off_by_one", "index_flip",
       "negate_cond", "remove_stmt", "remove_early_exit", "swap_args", "default_value", "kw_bool_flip", "return_bool_flip",
       "name_sibling", "attr_sibling", "drop_except", "break_continue"]


# ----------------------------------------------------------------------------- features
def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ok = [r for r in rows if "error" not in r]
    return pd.DataFrame(ok), len(rows) - len(ok)


def lg(x):
    return np.log1p(x)


def build_features(df, margin=True, graph_variant="ast"):
    """-> dict of DataFrames per feature group (columns numeric), following PREREG section 4"""
    X = pd.DataFrame(index=df.index)
    L0 = pd.DataFrame(index=df.index)
    L0["stmt_chars"] = lg(df.stmt_chars)
    L0["stmt_words"] = lg(df.stmt_words)
    L0["style_obs"] = (df["style"] == "observed").astype(float)
    L0["style_repro"] = (df["style"] == "repro").astype(float)
    L0["name_in_stmt"] = df.name_in_stmt.astype(float)
    L0["name_overlap"] = df.name_overlap.astype(float)
    L0["sym_eq_name"] = df.sym_eq_name.astype(float)
    if margin:
        tot = df.bm25_s_t.replace(0, np.nan)
        L0["bm25_rel"] = (df.bm25_s_g / tot).fillna(0.0)
        L0["bm25_margin"] = ((df.bm25_s_g - df.bm25_s_o) / tot).fillna(0.0)
    L1 = L0.copy()
    for op in OPS:
        L1["op_" + op] = df.ops.apply(lambda o: float(op in o))
    L1["pair"] = (df.kind == "pair").astype(float)
    L1["n_gold"] = df.n_gold.astype(float)
    L1["func_len"] = lg(df.func_len.fillna(df.func_len.median()))
    L1["file_len"] = lg(df.file_len.fillna(df.file_len.median()))
    L1["repo_n_defs"] = np.log(df.repo_n_defs.astype(float))
    L1["repo_n_files"] = np.log(df.repo_n_files.astype(float))
    L1["public"] = df.public.astype(float)
    L1["has_doc"] = df.has_doc.astype(float)
    pre = "nm_" if graph_variant == "nm" else ""

    def dist(col):
        v = df[col] if col in df else pd.Series(np.nan, index=df.index)
        na = v.isna()
        return v.clip(upper=6).fillna(7.0), na.astype(float)

    Gs = pd.DataFrame(index=df.index)
    Gt = pd.DataFrame(index=df.index)
    Gt["d_test"], Gt["d_test_na"] = dist("nm_d_test" if pre else "d_test")
    Gt["d_sym"], Gt["d_sym_na"] = dist("nm_d_sym" if pre else "d_sym")
    if not pre:
        Gt["d_sym_und"], Gt["d_sym_und_na"] = dist("d_sym_und")
    Gs["d_stmt"], Gs["d_stmt_na"] = dist("nm_d_stmt" if pre else "d_stmt")
    if not pre:
        Gs["n_stmt_cand"] = lg(df.n_stmt_cand.fillna(0))
    Gs["fan_in"] = lg(df["nm_fan_in" if pre else "fan_in"].fillna(0))
    Gs["fan_out"] = lg(df["nm_fan_out" if pre else "fan_out"].fillna(0))
    if not pre:
        Gt["fan_in_test"] = lg(df.fan_in_test.fillna(0))
        Gs["anc"] = lg(df.anc.fillna(0))
        Gs["desc"] = lg(df.desc.fillna(0))
        Gs["pagerank"] = np.log(df.pagerank.fillna(1.0).clip(lower=1e-9))
        Gt["n_tests_static"] = lg(df.n_tests_static.fillna(0))
        Gt["n_cover_dyn"] = lg(df.n_cover.fillna(0))
        Gt["n_cover_na"] = df.n_cover.isna().astype(float)
        Gs["module_depth"] = df.module_depth.fillna(0).astype(float)
        Gs["is_method"] = df.is_method.fillna(False).astype(float)
        Gs["nest_depth"] = df.nest_depth.fillna(0).astype(float)
    Gt["n_f2p"] = np.log(df.n_f2p.astype(float))
    return {"L0": L0, "L1": L1, "Gs": Gs, "Gt": Gt}


def combine(F, name):
    if name == "L0":
        return F["L0"]
    if name == "L1":
        return F["L1"]
    if name == "G":
        return pd.concat([F["Gs"], F["Gt"]], axis=1)
    if name == "Full":
        return pd.concat([F["L1"], F["Gs"], F["Gt"]], axis=1)
    if name == "L0+G":
        return pd.concat([F["L0"], F["Gs"], F["Gt"]], axis=1)
    if name == "L1+Gs":
        return pd.concat([F["L1"], F["Gs"]], axis=1)
    if name == "L1+Gt":
        return pd.concat([F["L1"], F["Gt"]], axis=1)
    raise KeyError(name)


# ----------------------------------------------------------------------------- models
def fit_predict(model, Xtr, ytr, gtr, Xte):
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).values, Xte.fillna(med).values
    if model == "M1":
        sc = StandardScaler().fit(Xtr)
        A, B = sc.transform(Xtr), sc.transform(Xte)
        best, bc = None, 1.0
        ngroups = len(set(gtr))
        if ngroups >= 5 and len(set(ytr)) > 1:
            gkf = GroupKFold(n_splits=5)
            losses = {}
            for C in CGRID:
                ll = 0.0
                for a, b in gkf.split(A, ytr, gtr):
                    if len(set(ytr[a])) < 2:
                        continue
                    m = LogisticRegression(C=C, max_iter=2000).fit(A[a], ytr[a])
                    ll += log_loss(ytr[b], m.predict_proba(A[b])[:, 1], labels=[0, 1]) * len(b)
                losses[C] = ll
            bc = min(losses, key=losses.get)
        m = LogisticRegression(C=bc, max_iter=3000).fit(A, ytr)
        return m.predict_proba(B)[:, 1]
    m = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=20,
                                   random_state=0).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def loro(X, y, groups, model):
    pred = np.zeros(len(y))
    for g in sorted(set(groups)):
        te = groups == g
        tr = ~te
        if len(set(y[tr])) < 2:
            pred[te] = y[tr].mean()
            continue
        pred[te] = fit_predict(model, X[tr].reset_index(drop=True) if hasattr(X, "reset_index") else X[tr], y[tr], groups[tr],
                               X[te].reset_index(drop=True) if hasattr(X, "reset_index") else X[te])
    return pred


def auc(y, p):
    """AUC via rank statistic (ties averaged)"""
    from scipy.stats import rankdata
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cluster_boot(y, preds, clusters, units=None, nboot=NBOOT, seed=SEED):
    """two-stage bootstrap: clusters (repos) then units within drawn clusters.  preds: {name: array}.  Returns {name: array(nboot)}"""
    rng = np.random.default_rng(seed)
    cl = sorted(set(clusters))
    idx_by = {c: np.where(clusters == c)[0] for c in cl}
    out = {k: np.empty(nboot) for k in preds}
    for b in range(nboot):
        pick = rng.choice(len(cl), size=len(cl), replace=True)
        ids = []
        for c in pick:
            ii = idx_by[cl[c]]
            ids.append(rng.choice(ii, size=len(ii), replace=True))
        ids = np.concatenate(ids)
        for k, p in preds.items():
            out[k][b] = auc(y[ids], p[ids])
    return out


def ci(a):
    a = a[~np.isnan(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def gain_stats(y, pa, pb, clusters, boots=None):
    """gain = AUC(pa) - AUC(pb)"""
    g0 = auc(y, pa) - auc(y, pb)
    bt = boots if boots is not None else cluster_boot(y, {"a": pa, "b": pb}, clusters)
    d = bt["a"] - bt["b"]
    lo, hi = ci(d)
    p_one = float((np.sum(d <= 0) + 1) / (len(d) + 1))
    return {"auc_a": float(auc(y, pa)), "auc_b": float(auc(y, pb)), "gain": float(g0), "ci": [lo, hi], "p_one_sided": p_one,
            "p_two_sided": min(1.0, 2 * p_one)}


def verdict(g):
    if g["gain"] < 0.03 or g["ci"][1] < 0.03:
        return "NULL"
    if g["ci"][0] > 0:
        return "SUPPORTED"
    return "INCONCLUSIVE (point >= 0.03, CI lower bound <= 0)"


# ----------------------------------------------------------------------------- main
def main():
    feats, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    df, n_err = load(feats)
    df = df.reset_index(drop=True)
    R = {"n_rows": int(len(df)), "n_error_rows": int(n_err)}
    R["gold_missing"] = int(df.gold_missing.sum())
    groups = df.repo.values.astype(str)
    # ---- targets
    T = {
        "T1_v6_func@5": (df.v6_func_rank.fillna(1e9) <= 5).astype(int).values,
        "T2_v6_func@1": (df.v6_func_rank.fillna(1e9) <= 1).astype(int).values,
        "T3_bm25_func@5": (df.bm25_func_rank.fillna(1e9) <= 5).astype(int).values,
        "T4_bm25_func@1": (df.bm25_func_rank.fillna(1e9) <= 1).astype(int).values,
        "T5_v6_file@3": (df.v6_file_rank.fillna(1e9) <= 3).astype(int).values,
    }
    R["target_rates"] = {k: float(v.mean()) for k, v in T.items()}
    F_m = build_features(df, margin=True)
    F_nm = build_features(df, margin=False)
    sets = ["L0", "L1", "G", "Full", "L0+G", "L1+Gs", "L1+Gt"]
    import pickle
    cache_f = out / "preds_cache.pkl"
    preds = pickle.load(open(cache_f, "rb")) if cache_f.exists() else {}

    def get_preds(tname, fs, model, margin):
        key = (tname, fs, model, margin)
        if key not in preds:
            X = combine(F_m if margin else F_nm, fs)
            preds[key] = loro(X, T[tname], groups, model)
            pickle.dump(preds, open(cache_f, "wb"))
        return preds[key]

    # ---- primary
    y = T["T1_v6_func@5"]
    pL1 = get_preds("T1_v6_func@5", "L1", "M1", True)
    pFull = get_preds("T1_v6_func@5", "Full", "M1", True)
    prim = gain_stats(y, pFull, pL1, groups)
    prim["verdict"] = verdict(prim)
    R["primary"] = prim
    print("PRIMARY", json.dumps(prim))
    # ---- AUC table by feature set / model / target
    auc_tab = {}
    boots_cache = {}
    for tname in T:
        margin = tname in ("T1_v6_func@5", "T2_v6_func@1", "T5_v6_file@3")
        auc_tab[tname] = {}
        for model in ("M1", "M2"):
            ps = {fs: get_preds(tname, fs, model, margin) for fs in sets}
            bt = cluster_boot(T[tname], ps, groups)
            boots_cache[(tname, model)] = bt
            auc_tab[tname][model] = {fs: {"auc": float(auc(T[tname], ps[fs])), "ci": ci(bt[fs])} for fs in sets}
            gains = {}
            for a, b in [("Full", "L1"), ("L0+G", "L0"), ("L1", "L0"), ("L1+Gs", "L1"), ("L1+Gt", "L1")]:
                d = bt[a] - bt[b]
                g = float(auc(T[tname], ps[a]) - auc(T[tname], ps[b]))
                p1 = float((np.sum(d <= 0) + 1) / (len(d) + 1))
                gains["%s-%s" % (a, b)] = {"gain": g, "ci": ci(d), "p_two_sided": min(1.0, 2 * p1)}
            auc_tab[tname][model]["gains"] = gains
    R["auc_table"] = auc_tab
    # primary bootstrap shares the same resamples (same seed, same clusters) -> identical to auc_tab entry
    # ---- POST-HOC (not pre-registered): v6 targets with the BM25-margin features removed from every set
    ph = {}
    for tname in ("T1_v6_func@5", "T2_v6_func@1", "T5_v6_file@3"):
        ps = {fs: get_preds(tname, fs, "M1", False) for fs in ("L0", "L1", "G", "Full", "L1+Gs", "L1+Gt")}
        bt = cluster_boot(T[tname], ps, groups)
        ent = {"auc": {fs: {"auc": float(auc(T[tname], ps[fs])), "ci": ci(bt[fs])} for fs in ps}, "gains": {}}
        for a, b in [("Full", "L1"), ("L1", "L0"), ("L1+Gs", "L1"), ("L1+Gt", "L1")]:
            d = bt[a] - bt[b]
            ent["gains"]["%s-%s" % (a, b)] = {"gain": float(auc(T[tname], ps[a]) - auc(T[tname], ps[b])), "ci": ci(d)}
        ph[tname] = ent
    R["posthoc_no_margin"] = ph
    # ---- Holm over the secondary family S = {T2,T3,T4,T5,A1}
    fam = {}
    for tname in ("T2_v6_func@1", "T3_bm25_func@5", "T4_bm25_func@1", "T5_v6_file@3"):
        fam[tname] = auc_tab[tname]["M1"]["gains"]["Full-L1"]
    # ---- agent targets
    runs = pd.DataFrame([json.loads(l) for l in open(REPO / "teacher/C_behaviour/per_run_table.jsonl", encoding="utf-8")])
    ag = {}
    sets_ag = {"A1": ("ab1", "base"), "A2": ("ab1", "adapter"), "A3": ("pilot1", "harvest")}
    base = df.set_index("iid")
    Fm_all = F_m
    for an, (src, arm) in sets_ag.items():
        sub = runs[(runs.src == src) & (runs.arm == arm)]
        ids = [t for t in sub.task if t in base.index]
        idx = [int(np.where(df.iid.values == t)[0][0]) for t in ids]
        yy = sub.set_index("task").loc[ids, "resolved"].astype(int).values
        gg = groups[idx]
        ent = {"n": len(ids), "rate": float(yy.mean()), "n_repos": int(len(set(gg)))}
        pr = {}
        for fs in ("L0", "L1", "G", "Full"):
            X = combine(Fm_all, fs).iloc[idx].reset_index(drop=True)
            pr[fs] = loro(X, yy, gg, "M1")
        bt = cluster_boot(yy, pr, gg)
        ent["auc"] = {fs: {"auc": float(auc(yy, pr[fs])), "ci": ci(bt[fs])} for fs in pr}
        d = bt["Full"] - bt["L1"]
        p1 = float((np.sum(d <= 0) + 1) / (len(d) + 1))
        ent["gain_Full-L1"] = {"gain": float(auc(yy, pr["Full"]) - auc(yy, pr["L1"])), "ci": ci(d), "p_two_sided": min(1.0, 2 * p1)}
        d2 = bt["G"] - bt["L0"]
        ent["gain_G-L0"] = {"gain": float(auc(yy, pr["G"]) - auc(yy, pr["L0"])), "ci": ci(d2)}
        # the localiser rank itself as a predictor (is localisation difficulty the mechanism?)
        yv6 = T["T1_v6_func@5"][idx]
        ent["auc_of_v6_hit_alone"] = float(auc(yy, yv6.astype(float)))
        ag[an] = ent
    # A4 pooled (rows stacked; clusters repo; run_set one-hot)
    rows_all, ys, gs, rs, tasks_ = [], [], [], [], []
    for an, (src, arm) in sets_ag.items():
        sub = runs[(runs.src == src) & (runs.arm == arm)]
        for t, r in zip(sub.task, sub.resolved):
            if t in base.index:
                rows_all.append(int(np.where(df.iid.values == t)[0][0]))
                ys.append(int(r))
                rs.append(an)
                tasks_.append(t)
    idx = np.array(rows_all)
    yy = np.array(ys)
    gg = groups[idx]
    prs = {}
    for fs in ("L1", "Full"):
        X = combine(Fm_all, fs).iloc[idx].reset_index(drop=True)
        for a in ("A1", "A2", "A3"):
            X["run_" + a] = [float(r == a) for r in rs]
        prs[fs] = loro(X, yy, gg, "M1")
    bt = cluster_boot(yy, prs, gg)
    d = bt["Full"] - bt["L1"]
    ag["A4"] = {"n_rows": int(len(yy)), "n_tasks": len(set(tasks_)), "gain_Full-L1": {"gain": float(auc(yy, prs["Full"]) - auc(yy, prs["L1"])), "ci": ci(d)},
                "auc": {fs: float(auc(yy, prs[fs])) for fs in prs}}
    R["agent"] = ag
    fam["A1"] = ag["A1"]["gain_Full-L1"]
    order = sorted(fam, key=lambda k: fam[k]["p_two_sided"])
    m = len(order)
    holm, prev = {}, 0.0
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * fam[k]["p_two_sided"])
        prev = max(prev, adj)
        holm[k] = prev
    R["secondary_holm"] = {k: {"gain": fam[k]["gain"], "ci": fam[k]["ci"], "p_two_sided": fam[k]["p_two_sided"], "holm_p": holm[k]} for k in fam}
    # ---- S1 stratified T1 gain
    strat = {}
    for name, mask in [("style_terse", df["style"] == "terse"), ("style_observed", df["style"] == "observed"), ("style_repro", df["style"] == "repro"),
                       ("heldout", df.split == "heldout"), ("train", df.split == "train"), ("single", df.kind == "single"), ("pair", df.kind == "pair")]:
        mk = mask.values
        yy = y[mk]
        if len(set(yy)) < 2:
            continue
        g = gain_stats(yy, pFull[mk], pL1[mk], groups[mk])
        g["n"] = int(mk.sum())
        strat[name] = g
    R["stratified_T1"] = strat
    # ---- S3 name-graph robustness (T1 gain with nm_ graph features)
    F_nmgraph = build_features(df, margin=True, graph_variant="nm")
    Xn = pd.concat([F_nmgraph["L1"], F_nmgraph["Gs"], F_nmgraph["Gt"]], axis=1)
    pn = loro(Xn, y, groups, "M1")
    R["S3_name_graph_T1"] = gain_stats(y, pn, pL1, groups)
    # ---- E1 dose-response
    dose = {}
    for col in ("d_test", "d_sym", "d_stmt"):
        v = df[col]
        bins = pd.cut(v.fillna(99), [-0.5, 0.5, 1.5, 2.5, 3.5, 98, 1000], labels=["0", "1", "2", "3", "4+", "NA"])
        dd = {}
        for tn in ("T1_v6_func@5", "T3_bm25_func@5"):
            g = pd.DataFrame({"b": bins, "y": T[tn]}).groupby("b", observed=False).y.agg(["mean", "count"])
            dd[tn] = {str(k): [float(r["mean"]) if r["count"] else None, int(r["count"])] for k, r in g.iterrows()}
        dose[col] = dd
    R["dose_raw"] = dose
    # matched / adjusted slope: logistic with strata dummies (repo x style x name_in_stmt x tertile) + ordinal distance (reachable only)
    adj = {}
    q = pd.qcut(df.stmt_words, 3, labels=False, duplicates="drop")
    strata = (df.repo.astype(str) + "|" + df["style"].astype(str) + "|" + df.name_in_stmt.astype(int).astype(str) + "|" + q.astype(str))
    for col in ("d_test", "d_sym", "d_stmt"):
        for tn in ("T1_v6_func@5", "T3_bm25_func@5"):
            adj[col + "|" + tn] = slope_fe(df, col, T[tn], strata, groups)
    R["dose_adjusted"] = adj
    # ---- E2 counterfactual
    cf = {}
    for vn in ("c1", "c2", "c12"):
        ch = df["cf_%s_changed" % vn].astype(bool).values
        for tn, col, orig in (("v6", "cf_%s_v6_func_rank" % vn, "v6_func_rank"), ("bm25", "cf_%s_bm25_func_rank" % vn, "bm25_func_rank")):
            o5 = (df[orig].fillna(1e9) <= 5).astype(float).values
            c5 = (df[col].fillna(1e9) <= 5).astype(float).values
            ent = {"n_changed": int(ch.sum()), "loss_all": float((o5 - c5).mean()), "loss_all_ci": boot_mean_ci((o5 - c5), groups),
                   "loss_changed": float((o5 - c5)[ch].mean()) if ch.sum() else None,
                   "loss_changed_ci": boot_mean_ci((o5 - c5)[ch], groups[ch]) if ch.sum() > 5 else None, "by_dist": {}}
            for col_d in ("d_test", "d_sym"):
                v = df[col_d].fillna(99).values
                for nm, mk in [("0-1", v <= 1), ("2", v == 2), ("3+", (v >= 3) & (v < 99)), ("NA", v >= 99)]:
                    mk2 = mk & ch
                    ent["by_dist"][col_d + ":" + nm] = [int(mk2.sum()), float((o5 - c5)[mk2].mean()) if mk2.sum() else None]
            cf.setdefault(vn, {})[tn] = ent
    R["counterfactual"] = cf
    # ---- leakage / observability checks
    leak = {}
    leak["fan_out_S_vs_pristine_equal_share"] = float((df.fan_out.round(6) == df.fan_out_pristine.round(6)).mean())
    leak["fan_out_mean_abs_diff"] = float((df.fan_out - df.fan_out_pristine).abs().mean())
    sym = df[df.symbol]
    leak["symbol_present_share"] = float(df.symbol.mean())
    leak["symbol_in_stmt_by_style"] = {s: float(g.sym_in_stmt.mean()) for s, g in sym.groupby("style")}
    leak["symbol_in_test_body_share"] = float(sym.sym_in_test_body.mean()) if len(sym) else None
    leak["d_test_na_share"] = float(df.d_test.isna().mean())
    leak["f2p_found_share"] = float((df.n_f2p_found / df.n_f2p).mean())
    leak["graph_resolved_share_direct"] = float((df.g_n_direct / df.g_n_calls.clip(lower=1)).mean())
    leak["graph_fallback_share"] = float((df.g_n_fallback / df.g_n_calls.clip(lower=1)).mean())
    leak["graph_unresolved_share"] = float((df.g_n_unresolved / df.g_n_calls.clip(lower=1)).mean())
    leak["n_cover_na_share"] = float(df.n_cover.isna().mean())
    leak["per_repo_auc_T1_Full"] = {r: float(auc(y[groups == r], pFull[groups == r])) for r in sorted(set(groups)) if len(set(y[groups == r])) > 1}
    leak["per_repo_auc_T1_L1"] = {r: float(auc(y[groups == r], pL1[groups == r])) for r in sorted(set(groups)) if len(set(y[groups == r])) > 1}
    R["leakage"] = leak
    # ---- correlations among graph features and with lexical ones (descriptive)
    G_all = combine(F_m, "G")
    R["corr_with_T1"] = {c: float(np.corrcoef(G_all[c].fillna(G_all[c].median()), y)[0, 1]) for c in G_all.columns
                         if G_all[c].std() > 0}
    (out / "results.json").write_text(json.dumps(R, indent=1, default=float), encoding="utf-8")
    make_tables(R, out)
    try:
        make_figs(df, R, out, T)
    except Exception as e:
        print("figures failed", repr(e))
    print("wrote", out / "results.json")


def boot_mean_ci(v, clusters, nboot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    cl = sorted(set(clusters))
    idx_by = {c: np.where(clusters == c)[0] for c in cl}
    ms = []
    for _ in range(nboot):
        pick = rng.choice(len(cl), size=len(cl), replace=True)
        ids = np.concatenate([rng.choice(idx_by[cl[c]], size=len(idx_by[cl[c]]), replace=True) for c in pick])
        ms.append(v[ids].mean())
    return [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]


def slope_fe(df, col, yv, strata, groups, nboot=300):
    """log-odds per hop with stratum fixed effects (reachable tasks only, strata with >1 distinct distance and both outcomes)"""
    v = df[col]
    ok = v.notna().values
    d = v.clip(upper=6).values.astype(float)
    S = strata.values
    keep = np.zeros(len(df), bool)
    for s in set(S[ok]):
        m = ok & (S == s)
        if m.sum() >= 4 and len(set(d[m])) > 1 and len(set(yv[m])) > 1:
            keep |= m
    if keep.sum() < 30:
        return {"n": int(keep.sum()), "note": "too few informative strata"}

    def fit(idx):
        sub_s = pd.get_dummies(pd.Series(S[idx]), drop_first=True).astype(float).values
        X = np.column_stack([d[idx], sub_s]) if sub_s.shape[1] else d[idx].reshape(-1, 1)
        if len(set(yv[idx])) < 2:
            return float("nan")
        try:
            m = LogisticRegression(C=50.0, max_iter=3000).fit(X, yv[idx])
            return float(m.coef_[0][0])
        except Exception:
            return float("nan")

    idx0 = np.where(keep)[0]
    b0 = fit(idx0)
    rng = np.random.default_rng(SEED)
    cl = sorted(set(groups[idx0]))
    byc = {c: idx0[groups[idx0] == c] for c in cl}
    bs = []
    for _ in range(nboot):
        pick = rng.choice(len(cl), size=len(cl), replace=True)
        ids = np.concatenate([byc[cl[c]] for c in pick])
        bs.append(fit(ids))
    bs = np.array([b for b in bs if not np.isnan(b)])
    return {"n": int(keep.sum()), "log_odds_per_hop": b0, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) > 20 else None}


# ----------------------------------------------------------------------------- output
def f3(x):
    return "%.3f" % x


def make_tables(R, out):
    L = []
    p = R["primary"]
    L.append("## Primary\n\nT1 (v6 func@5), M1, leave-one-repo-out, n=%d, positives %.1f%%\n" % (R["n_rows"], 100 * R["target_rates"]["T1_v6_func@5"]))
    L.append("AUC(Full) %s, AUC(L1) %s, gain %s [%s, %s] -> **%s**\n" % (f3(p["auc_a"]), f3(p["auc_b"]), f3(p["gain"]), f3(p["ci"][0]), f3(p["ci"][1]), p["verdict"]))
    L.append("## AUC by feature set\n")
    for tn, d in R["auc_table"].items():
        for model in ("M1", "M2"):
            L.append("**%s %s**\n\n| feature set | AUC | 95%% CI |\n|---|---|---|" % (tn, model))
            for fs, v in d[model].items():
                if fs == "gains":
                    continue
                L.append("| %s | %s | [%s, %s] |" % (fs, f3(v["auc"]), f3(v["ci"][0]), f3(v["ci"][1])))
            L.append("\n| gain | value | 95% CI | p (2-sided) |\n|---|---|---|---|")
            for k, v in d[model]["gains"].items():
                L.append("| %s | %s | [%s, %s] | %.3f |" % (k, f3(v["gain"]), f3(v["ci"][0]), f3(v["ci"][1]), v["p_two_sided"]))
            L.append("")
    L.append("## Secondary family (Holm)\n\n| target | gain Full-L1 | 95% CI | p | Holm p |\n|---|---|---|---|---|")
    for k, v in R["secondary_holm"].items():
        L.append("| %s | %s | [%s, %s] | %.3f | %.3f |" % (k, f3(v["gain"]), f3(v["ci"][0]), f3(v["ci"][1]), v["p_two_sided"], v["holm_p"]))
    L.append("\n## Agent targets (M1)\n\n| target | n | resolved | AUC L0 | AUC L1 | AUC G | AUC Full | gain Full-L1 [CI] |\n|---|---|---|---|---|---|---|---|")
    for k, v in R["agent"].items():
        if k == "A4":
            L.append("| A4 pooled | %d rows / %d tasks | | | %s | | %s | %s [%s, %s] |" % (v["n_rows"], v["n_tasks"], f3(v["auc"]["L1"]), f3(v["auc"]["Full"]), f3(v["gain_Full-L1"]["gain"]), f3(v["gain_Full-L1"]["ci"][0]), f3(v["gain_Full-L1"]["ci"][1])))
        else:
            a = v["auc"]
            L.append("| %s | %d | %.2f | %s | %s | %s | %s | %s [%s, %s] |" % (k, v["n"], v["rate"], f3(a["L0"]["auc"]), f3(a["L1"]["auc"]), f3(a["G"]["auc"]), f3(a["Full"]["auc"]),
                                                                       f3(v["gain_Full-L1"]["gain"]), f3(v["gain_Full-L1"]["ci"][0]), f3(v["gain_Full-L1"]["ci"][1])))
    L.append("\n## Stratified T1 gain (Full - L1)\n\n| stratum | n | gain | 95% CI |\n|---|---|---|---|")
    for k, v in R["stratified_T1"].items():
        L.append("| %s | %d | %s | [%s, %s] |" % (k, v["n"], f3(v["gain"]), f3(v["ci"][0]), f3(v["ci"][1])))
    s3 = R["S3_name_graph_T1"]
    L.append("\nS3 name-graph variant: gain %s [%s, %s]\n" % (f3(s3["gain"]), f3(s3["ci"][0]), f3(s3["ci"][1])))
    L.append("## Dose-response (raw hit rate by distance bin; [rate, n])\n")
    for col, dd in R["dose_raw"].items():
        for tn, v in dd.items():
            L.append("- %s / %s: %s" % (col, tn, json.dumps(v)))
    L.append("\n## Dose-response, stratum-fixed-effects log-odds per hop\n")
    for k, v in R["dose_adjusted"].items():
        L.append("- %s: %s" % (k, json.dumps(v)))
    L.append("\n## Counterfactual (hit@5 loss = original - neutralised)\n")
    for vn, d in R["counterfactual"].items():
        for tn, e in d.items():
            L.append("- %s %s: n_changed=%d, loss_all %s CI %s, loss_changed %s CI %s, by_dist %s" % (vn, tn, e["n_changed"], f3(e["loss_all"]), e["loss_all_ci"], e["loss_changed"], e["loss_changed_ci"], json.dumps(e["by_dist"])))
    L.append("\n## Leakage / observability checks\n\n```\n" + json.dumps(R["leakage"], indent=1) + "\n```")
    L.append("\n## Univariate correlation of graph features with T1\n\n" + json.dumps(R["corr_with_T1"], indent=1))
    (out / "tables.md").write_text("\n".join(L), encoding="utf-8")


def make_figs(df, R, out, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, col in zip(ax, ("d_test", "d_sym", "d_stmt")):
        for tn, c in (("T1_v6_func@5", "#1f77b4"), ("T3_bm25_func@5", "#d62728")):
            d = R["dose_raw"][col][tn]
            labs = list(d)
            a.plot(range(len(labs)), [d[k][0] if d[k][0] is not None else np.nan for k in labs], "o-", color=c, label=tn.split("_", 1)[1])
        a.set_xticks(range(len(labs)))
        a.set_xticklabels(labs)
        a.set_xlabel("directed call distance (%s)" % col)
        a.set_ylim(0, 1)
    ax[0].set_ylabel("hit rate (func@5)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / "dose_response.png", dpi=140)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    names = ["L0", "L1", "G", "L1+Gs", "L1+Gt", "Full"]
    for k, tn in enumerate(("T1_v6_func@5", "T3_bm25_func@5", "T2_v6_func@1")):
        d = R["auc_table"][tn]["M1"]
        ys = [d[n]["auc"] for n in names]
        err = [[d[n]["auc"] - d[n]["ci"][0] for n in names], [d[n]["ci"][1] - d[n]["auc"] for n in names]]
        ax.errorbar(np.arange(len(names)) + 0.12 * k, ys, yerr=err, fmt="o", capsize=2, label=tn)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("pooled LORO AUC")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "auc_by_set.png", dpi=140)


if __name__ == "__main__":
    main()
