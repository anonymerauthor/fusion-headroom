"""One consistent set of estimates per committee instance.

headroom = L_best - H_all, H_all = tightest available bound (results/acct_<regime>.csv);
cal = L_best - H_best (logistic recalibration of the best member, learning-curve corrected for n=128/512);
ci  = H_best - H_all, so that headroom = cal + ci holds exactly in every table.
Output: results/unified_<regime>.csv
"""
import json, os
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EST = {"full": "lr", "n512": "lr_lc", "n128": "lr_lc"}
for r in ("n128", "n512", "full"):
    a = pd.read_csv(os.path.join(ROOT, "results", f"acct_{r}.csv")).set_index(["dataset", "fold"])
    d = pd.read_csv(os.path.join(ROOT, "results", f"real_{r}_{EST[r]}.csv")).set_index(["dataset", "fold"])
    j = d.join(a[["L_best", "H_all", "headroom"]].add_suffix("_acct"), how="inner")
    Lb = j.apply(lambda x: json.loads(x.L_json)[x.k_star], axis=1)
    Hb = j.apply(lambda x: json.loads(x.H_json)[x.k_star], axis=1)
    assert (Lb - j.L_best_acct).abs().max() < 1e-5, "best-member losses differ between runs"
    out = pd.DataFrame({"split": j.split, "n": j.n, "C": j.C, "L_best": j.L_best_acct, "H_best": Hb,
                        "H_all": j.H_all_acct, "headroom": j.headroom_acct})
    out["cal"] = out.L_best - out.H_best
    out["ci"] = out.H_best - out.H_all
    out = out.join(j[[c for c in j.columns if c.startswith("div_")]])
    out.to_csv(os.path.join(ROOT, "results", f"unified_{r}.csv"))
    t = out[out.split == "test"].groupby(level=0).mean(numeric_only=True)
    print(f"{r}: test data sets {len(t)} | headroom {t.headroom.mean():.3f} = cal {t.cal.mean():.3f} + ci {t.ci.mean():.3f}"
          f" | all 63: headroom {out.groupby(level=0).headroom.mean().mean():.3f}")
