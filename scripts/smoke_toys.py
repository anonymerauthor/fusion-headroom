"""Smoke test: ground truth vs. estimates on the four toy committees (C=2 and C=3)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fh.synth import toy
from fh.info import decompose

rng = np.random.default_rng(42)
for C in (2, 3):
    for kind in ("redundant", "noise", "synergy", "complement"):
        g = toy(kind, C)
        t = g.truth([(0,), (1,), (0, 1)], N=200_000)
        ks = int(np.argmin(t["L"]))
        true_head = t["L"][ks] - t[(0, 1)]
        true_ci = t[(ks,)] - t[(0, 1)]
        P, y, _ = g.sample(2000, rng)
        agree = (P[:, 0].argmax(1) == P[:, 1].argmax(1)).mean()
        d = decompose(P, y)
        print(f"C={C} {kind:10s} agree={agree:.2f} | true I1={t['H_y']-t[(0,)]:.3f} I2={t['H_y']-t[(1,)]:.3f} "
              f"CI={true_ci:.3f} head={true_head:.3f} | est CI={d['ci']:.3f} head={d['headroom']:.3f}")
