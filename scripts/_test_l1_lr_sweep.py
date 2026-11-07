"""
scripts/_test_l1_lr_sweep.py

Does the predictor's deficit come from the learning rate or from the model?

The comparison against a "next step equals the last step" baseline came out badly
for the model: 0.1167 against 0.0518. Two explanations are possible.

    learning rate too small   the weights have not settled, so the model is
                              still worse than a trivial rule that needs no
                              learning at all
    model wrong               the weights could settle given time, but even
                              settled a linear map of the previous projection
                              cannot describe this sequence

Distinguishing them is cheap: run the same comparison at several fractions of
the stability limit and see whether the gap closes. If it closes, the layer needs
a faster rate. If it stays open at every rate, a linear map is not the right
form and no amount of training will help.

All three are scored in the same 17-dimensional projection, over the same stream.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_l1_lr_sweep.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.runtime.process_1_fast import FHNNetwork, L1Predictor


def trial(X, Z, fraction, steps=None):
    pred = L1Predictor(input_dim=X.shape[1], stability_fraction=fraction)
    base = pred.set_baseline(X[:3000])
    # Recompute the projection under the calibrated width.
    Z = X @ pred.proj

    n = steps or Z.shape[0]
    errs = []
    prev = []
    for i in range(1, n):
        pred.step(X[i])
        errs.append(float(np.linalg.norm(pred.last_error)))
        prev.append(float(np.linalg.norm(Z[i] - Z[i - 1])))

    h = len(errs) // 2
    m = float(np.mean(errs[h:]))
    p = float(np.mean(prev[h:]))
    return {
        "lr": pred.lr,
        "dim": pred.proj_dim,
        "model": m,
        "previous": p,
        "ratio": m / p if p > 0 else float("inf"),
    }


def main():
    net = FHNNetwork(seed=42)
    vecs = []
    for _ in range(8000):
        vecs.append(net.step()["E_rate_vec"])
    X = np.asarray(vecs, dtype=float)

    print(f"{'frac':>6} {'lr':>9} {'dim':>4} {'model':>9} {'previous':>9} "
          f"{'ratio':>7}")
    rows = []
    for frac in (0.05, 0.1, 0.25, 0.5, 0.9, 1.5):
        # The gradient update is dv = lr * outer(error, prev), applied to
        # W_pred; the stability limit applies to the effective step, and the
        # projection is orthonormalised by the sqrt normalisation below, so a
        # fraction above 1 is still meaningful arithmetic even though it risks
        # oscillation.
        r = trial(X, None, frac)
        rows.append((frac, r))
        print(f"{frac:>6.2f} {r['lr']:>9.5f} {r['dim']:>4} {r['model']:>9.4f} "
              f"{r['previous']:>9.4f} {r['ratio']:>7.2f}")

    print()
    print("=" * 58)
    best = min(rows, key=lambda t: t[1]["ratio"])
    print(f"best ratio: {best[1]['ratio']:.2f} at fraction {best[0]}")
    if best[1]["ratio"] < 0.95:
        print("VERDICT: at some learning rate the model beats the previous-step "
              "baseline. The deficit was training speed, not the model form.")
    else:
        print("VERDICT: no learning rate closes the gap. The previous step alone "
              "predicts better than a linear map of it, at every rate tried. "
              "A linear map of one step back is the wrong form for this signal.")


if __name__ == "__main__":
    main()
