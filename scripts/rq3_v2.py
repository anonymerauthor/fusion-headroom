"""Prediction of realised gains (test data sets only, unified estimates).

tab_rq3.tex       rho(headroom / complementarity / best classic diversity / best label-free diversity, realised gain)
                  with paired bootstrap CIs of rho(headroom) - |rho(best diversity)| and Holm-adjusted p-values.
tab_baselines.tex rule-specific and label-using baselines: the rule's own OOF CV gain (stacking for rules
                  that have none), best-member loss, loss spread, oracle accuracy, recalibration term; partial correlation
                  of the headroom with the realised gain given the stacking CV gain.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fh.data import RULE_RESULTS, iter_folds, load_fold  # noqa: E402

SUF = {"n128": "_n128", "n512": "_n512", "full": ""}
RULES = [("stack_LR", "Stacking"), ("greedy_ES", "Greedy ES"), ("NE_stack", "Neural ens."),
         ("pretrained", "Pretrained"), ("best_rule", "Best of 13 (oracle)")]
OWN = {"stack_LR": "stack_LR_gain", "greedy_ES": "greedy_ES_gain"}
LF = ["div_ambiguity_geo", "div_js_spread", "div_pred_disagreement"]
REG = [("full", "Full data"), ("n512", "$n=512$"), ("n128", "$n=128$")]
rng = np.random.default_rng(42)


def realised(r):
    fs = [f for sp in ("dev", "test") for k in ("cpu", "gpu", "pool", "pretrained")
          for f in glob.glob(os.path.join(RULE_RESULTS, f"{sp}_{k}_committees{SUF[r]}.csv"))]
    w = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["dataset", "fold", "method"], keep="last") \
        .pivot_table(index=["dataset", "fold"], columns="method", values="logloss")
    g = pd.DataFrame({c: w["single_best"] - w[c] for c in w.columns if c != "single_best" and not c.startswith("TabPFN")})
    g["best_rule"] = g.max(1)
    return g


def oracle(r):
    vals = {}
    for meta, fold, path in iter_folds("test", r):
        z = load_fold(path)
        vals[(meta["name"], fold)] = float((z["P_oof"].argmax(-1) == z["y_tr"].astype(int)[:, None]).any(1).mean())
    s = pd.Series(vals, name="oracle_acc"); s.index.names = ["dataset", "fold"]; return s


def rho(a, b):
    return spearmanr(a, b, nan_policy="omit").correlation


def pcor(x, y, z):
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z)); x, y, z = rankdata(x[m]), rankdata(y[m]), rankdata(z[m])
    ex = x - np.polyval(np.polyfit(z, x, 1), z); ey = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(ex, ey)[0, 1])


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, k in enumerate(o):
        run = max(run, min(1, (len(p) - i) * p[k])); adj[k] = run
    return adj


T2, T3, rec = [], [], []
for r, lab in REG:
    u = pd.read_csv(os.path.join(ROOT, "results", f"unified_{r}.csv")).set_index(["dataset", "fold"])
    real = pd.read_csv(os.path.join(ROOT, "results", f"real_{r}_{'lr' if r == 'full' else 'lr_lc'}.csv")).set_index(["dataset", "fold"])
    u["loss_spread"] = real.L_json.map(lambda s: max(json.loads(s).values()) - min(json.loads(s).values()))
    a = pd.read_csv(os.path.join(ROOT, "results", f"acct_{r}.csv")).set_index(["dataset", "fold"])
    j = u.join(a[["stack_LR_gain", "greedy_ES_gain"]]).join(realised(r), how="inner").join(oracle(r))
    m = j[j.split == "test"].groupby(level=0).mean(numeric_only=True)
    divs = [c for c in m.columns if c.startswith("div_")]
    for t, tl in RULES:
        bc = max([c for c in divs if c not in LF], key=lambda c: abs(rho(m[c], m[t])))
        bl = max(LF, key=lambda c: abs(rho(m[c], m[t])))
        bd = bc if abs(rho(m[bc], m[t])) >= abs(rho(m[bl], m[t])) else bl
        diffs = []
        for _ in range(2000):
            ix = rng.integers(0, len(m), len(m)); mb = m.iloc[ix]
            diffs.append(rho(mb.headroom, mb[t]) - abs(rho(mb[bd], mb[t])))
        diffs = np.array(diffs)
        rec.append(dict(regime=r, lab=lab, rule=tl, n=len(m), head=rho(m.headroom, m[t]), ci=rho(m.ci, m[t]),
                        cl=abs(rho(m[bc], m[t])), cln=bc[4:], lf=abs(rho(m[bl], m[t])), lfn=bl[4:],
                        d=rho(m.headroom, m[t]) - abs(rho(m[bd], m[t])), lo=np.nanpercentile(diffs, 2.5),
                        hi=np.nanpercentile(diffs, 97.5), p=min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))))
        own = OWN.get(t, "stack_LR_gain")
        simple = {k: abs(rho(m[k], m[t])) for k in ("L_best", "loss_spread", "oracle_acc", "cal")}
        T3.append(dict(regime=r, lab=lab, rule=tl, own=rho(m[own], m[t]), own_name="own" if t in OWN else "stack",
                       head=rho(m.headroom, m[t]), partial=pcor(m.headroom.values, m[t].values, m[own].values),
                       best_simple=max(simple.values()), simple_name=max(simple, key=simple.get)))
R = pd.DataFrame(rec); R["p_holm"] = holm(R.p.values)
NM = {"difficulty": "difficulty", "entropy_E": "entropy", "GD": "gen.\\ div.", "js_spread": "JS spread", "KW": "KW",
      "pred_disagreement": "pred.\\ disagr.", "ambiguity_geo": "ambiguity", "double_fault": "double fault",
      "kappa": "$\\kappa$", "Q": "$Q$", "rho": "$\\rho$", "disagreement": "disagr.", "CFD": "CFD"}
L = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccc@{}}", r"\toprule",
     r"Regime & Realised gain of & Headroom & Compl. & Best classic div. & Best label-free div. & $\Delta\rho$ [95\% CI] & $p_{\mathrm{Holm}}$ \\",
     r"\midrule"]
for r, lab in REG:
    x = R[R.regime == r]
    for i, row in enumerate(x.itertuples()):
        L.append(f"{(lab + f' ({row.n})') if i == 0 else ''} & {row.rule} & {row.head:.2f} & {row.ci:.2f} & "
                 f"{row.cl:.2f} ({NM[row.cln]}) & {row.lf:.2f} ({NM[row.lfn]}) & ${row.d:+.2f}$ [{row.lo:+.2f}, {row.hi:+.2f}] & "
                 f"{row.p_holm:.2f}{'$^{*}$' if row.p_holm < 0.05 else ''} \\\\")
    L.append(r"\midrule")
L = L[:-1] + [r"\bottomrule", r"\end{tabular*}"]
open(os.path.join(ROOT, "outputs", "tables", "tab_rq3.tex"), "w").write("\n".join(L) + "\n")
S = pd.DataFrame(T3)
SN = {"L_best": "best loss", "loss_spread": "loss spread", "oracle_acc": "oracle acc.", "cal": "recal. term"}
B = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccc@{}}", r"\toprule",
     r"Regime & Realised gain of & CV gain of a fitted rule$^\dagger$ & Best label-using predictor & Headroom & Partial $\rho$ (headroom $\mid$ CV gain) \\",
     r"\midrule"]
for r, lab in REG:
    x = S[S.regime == r]
    for i, row in enumerate(x.itertuples()):
        B.append(f"{lab if i == 0 else ''} & {row.rule} & {row.own:.2f} ({row.own_name}) & {row.best_simple:.2f} ({SN[row.simple_name]}) & "
                 f"{row.head:.2f} & {row.partial:+.2f} \\\\")
    B.append(r"\midrule")
B = B[:-1] + [r"\bottomrule", r"\end{tabular*}"]
open(os.path.join(ROOT, "outputs", "tables", "tab_baselines.tex"), "w").write("\n".join(B) + "\n")
pd.set_option("display.width", 250)
print(R[["regime", "rule", "head", "ci", "cl", "cln", "lf", "lfn", "d", "lo", "hi", "p_holm"]].round(2).to_string())
print(S.round(2).to_string())
