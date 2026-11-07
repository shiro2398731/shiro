"""
scripts/_test_l1_features.py

Which feature set is better, with everything else held fixed?

An earlier comparison changed the projection width between conditions without
noticing, so the "previous step" baseline moved from 0.052 to 0.041 and the
comparison measured the baseline rather than the model.

Here the projection is built once with a fixed seed and fixed width, and only the
feature map changes: lag (how many past steps) and power (up to which power).
The learning rate is fixed at the value derived earlier.

Scoring is on the second half of the stream, so every condition gets the same
targets.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_l1_features.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.runtime.process_1_fast import FHNNetwork, L1Predictor

PROJ = 17
SEED = 7
LR = 0.0118


def build(proj, lag, power):
    p = L1Predictor(input_dim=100, proj_dim=proj, lag=lag, power=power, lr=LR)
    p.proj = np.random.default_rng(SEED).normal(0, 1, (100, proj)) / np.sqrt(100)
    p.n_feat = proj * lag * power
    p.W_pred = np.zeros((p.n_feat, proj))
    p._z_hist = []
    return p


def run(X, lag, power):
    p = build(PROJ, lag, power)
    p._ridge_fit(X[:12000], PROJ)
    Z = X @ p.proj
    errs = []
    for i in range(1, Z.shape[0]):
        p.step(X[i])
        errs.append(float(np.linalg.norm(p.last_error)))
    E = np.asarray(errs)
    prev = np.linalg.norm(Z[1:] - Z[:-1], axis=1)
    h = len(E) // 2
    return float(E[h:].mean()), float(prev[h:].mean())


def main():
    net = FHNNetwork(seed=42)
    X = np.asarray([net.step()["E_rate_vec"] for _ in range(60000)])

    print(f"projection dim {PROJ}, seed {SEED}, lr {LR}, warm-up 12000")
    print()
    print(f"{'lag':>4} {'power':>6} {'feat':>5} {'online':>9} {'prev':>9} {'ratio':>7}")
    rows = []
    for lag, power in ((1, 1), (2, 1), (4, 1), (4, 2), (4, 3), (8, 2), (8, 3)):
        e, b = run(X, lag, power)
        r = e / b if b > 0 else float("inf")
        rows.append((lag, power, PROJ * lag * power, e, b, r))
        print(f"{lag:>4} {power:>6} {PROJ * lag * power:>5} {e:>9.4f} "
              f"{b:>9.4f} {r:>7.3f}")

    print()
    print("=" * 55)
    best = min(rows, key=lambda t: t[5])
    lin = [r for r in rows if r[1] == 1 and r[0] == 1][0]
    print(f"best      : lag={best[0]} power={best[1]} ratio={best[5]:.3f}")
    print(f"plain     : lag=1  power=1 ratio={lin[5]:.3f}")
    gain = (lin[5] - best[5]) / lin[5]
    if best[5] < lin[5]:
        print(f"VERDICT: the best feature set improves on the plain predictor by "
              f"{gain:.1%}. Worth adopting.")
    else:
        print("VERDICT: no feature set beats the plain predictor.")


if __name__ == "__main__":
    main()
