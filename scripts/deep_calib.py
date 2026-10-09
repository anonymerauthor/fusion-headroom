"""Is the complementarity share of deep committees an artefact of a weak single-member calibrator?

For the seed ensemble, the architecture committee and the full pool (fusion pool of 10k images) we bound
H(Y | P_k*) with (i) the structured log pool used in the paper (temperature + class intercepts) and (ii) Dirichlet
calibration = multinomial logistic regression on the C log-probabilities of the best member (full C x C matrix,
L2 chosen by inner CV), and H(Y | P) with (i) the log pool and (ii) multinomial LR on all members' log-probabilities.
Every bound is valid; we report the tightest and the resulting split. Output: results/deep_calib.csv"""
import json, os, re, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fh.info as info  # noqa: E402
from deep_analyze import load_pool, HETERO  # noqa: E402

info.FEAT_EPS = 1e-6
rows = []
for ds in ("cifar10", "cifar100"):
    pool, y, _ = load_pool(ds)
    names = sorted(pool)
    comms = {"seeds": [n for n in names if re.fullmatch(r"resnet18_s\d+", n)],
             "hetero": [n for n in HETERO if n in names], "all": names}
    for cname, mem in comms.items():
        P = np.stack([pool[m][0] for m in mem], 1)
        K, C = P.shape[1], P.shape[2]
        L = np.array([info.nll(P[:, k], y) for k in range(K)]); ks = int(np.argmin(L))
        r = {"dataset": ds, "committee": cname, "K": K, "L_best": L[ks]}
        r["Hk_logpool"] = info.cond_entropy(P, y, [ks], C, model="logpool")
        r["Hk_dirichlet"] = info.cond_entropy(P, y, [ks], C, model="lr")
        r["H_logpool"] = info.cond_entropy(P, y, range(K), C, model="logpool")
        r["H_lr"] = info.cond_entropy(P, y, range(K), C, model="lr") if K * C <= 1500 else np.nan
        for tag, hk, hall in (("paper", r["Hk_logpool"], r["H_logpool"]),
                              ("tightest", min(r["Hk_logpool"], r["Hk_dirichlet"]),
                               np.nanmin([r["H_logpool"], r["H_lr"]]))):
            r[f"head_{tag}"] = L[ks] - hall; r[f"cal_{tag}"] = L[ks] - hk; r[f"ci_{tag}"] = hk - hall
        rows.append(r); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "deep_calib.csv"), index=False)
