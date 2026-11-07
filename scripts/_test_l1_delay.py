"""
scripts/_test_l1_delay.py

How much history does the prediction need before it beats "next equals last"?

The predictor currently takes one projected point and produces the next. That is
a map from R^k to R^k with no notion of recent motion, and it was measured to at
best tie the trivial rule of predicting that nothing changes.

Takens' theorem says the missing thing is time: a single observed quantity,
sampled at several delays, spans the attractor even when the full state cannot be
observed. The oscillator's state is (v, w); only v is exposed. The theorem says a
window of v is enough, and that the number of delays needed is set by the
attractor's dimension -- not by preference.

Rather than compute that dimension and trust the arithmetic, this measures the
usefulness of each window length directly:

    for k = 1, 2, 3, ... build [z(t-1), ..., z(t-k)] and fit a linear map to z(t)
    compare its error with predicting z(t) = z(t-1)

The window that first beats the trivial rule is the smallest one carrying enough
information. Anything larger is not needed for this question.

The map is fit by ridge regression on the first half and scored on the second, so
the score is out of sample. Fitting and scoring on the same data would let a
large window memorise its way past the baseline.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_l1_delay.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.runtime.process_1_fast import FHNNetwork, L1Predictor


def build_windows(Z, k):
    """Stack the last k samples into each row."""
    n = Z.shape[0]
    X = np.empty((n - k, k * Z.shape[1]), dtype=float)
    for i in range(k):
        X[:, i * Z.shape[1]:(i + 1) * Z.shape[1]] = Z[k - 1 - i:n - 1 - i]
    Y = Z[k:]
    return X, Y


def main():
    net = FHNNetwork(seed=42)
    vecs = []
    for _ in range(12000):
        vecs.append(net.step()["E_rate_vec"])
    Xraw = np.asarray(vecs, dtype=float)

    # Project exactly as the predictor does, with its calibrated width.
    pred = L1Predictor(input_dim=Xraw.shape[1])
    base = pred.set_baseline(Xraw[:4000])
    Z = Xraw @ pred.proj
    print(f"projection: dim={pred.proj_dim} "
          f"({base['variance_explained']:.1%} variance), lr={pred.lr:.4f}")
    print()

    print(f"{'k':>3} {'inputs':>7} {'model':>9} {'previous':>9} {'ratio':>7}")
    results = []
    for k in (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32):
        X, Y = build_windows(Z, k)
        n = X.shape[0]
        h = n // 2

        Xtr, Ytr = X[:h], Y[:h]
        Xte, Yte = X[h:], Y[h:]

        # ridge, solved in closed form
        lam = 1e-3 * max(float(np.trace(Xtr.T @ Xtr)) / Xtr.shape[1], 1e-12)
        W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]),
                            Xtr.T @ Ytr)
        pred_err = np.linalg.norm(Yte - Xte @ W, axis=1)

        # previous-step baseline: the last block of the window is z(t-1)
        last = Xte[:, :Z.shape[1]]
        base_err = np.linalg.norm(Yte - last, axis=1)

        ratio = pred_err.mean() / base_err.mean()
        results.append((k, X.shape[1], pred_err.mean(), base_err.mean(), ratio))
        print(f"{k:>3} {X.shape[1]:>7} {pred_err.mean():>9.4f} "
              f"{base_err.mean():>9.4f} {ratio:>7.3f}")

    print()
    print("=" * 55)
    winners = [r for r in results if r[4] < 0.95]
    if winners:
        first = winners[0]
        print(f"VERDICT: a window of {first[0]} step(s) beats the previous-step "
              f"rule by {(1 - first[4]):.1%}. History is what the predictor "
              f"was missing.")
        best = min(results, key=lambda r: r[4])
        print(f"best ratio {best[4]:.3f} at k={best[0]}")
    else:
        print("VERDICT: no window length beats the previous-step rule. The "
              "signal may be smooth enough that history adds nothing, or the "
              "linear form is still wrong.")


if __name__ == "__main__":
    main()
