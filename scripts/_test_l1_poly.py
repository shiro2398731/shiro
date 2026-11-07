"""
scripts/_test_l1_poly.py

Does adding the nonlinear terms the oscillator actually has close the gap?

The oscillator's equation is

    dv = v - v^3 / 3 - w + I

The nonlinearity is cubic in the state. A predictor that only forms linear
combinations of the state cannot represent that term, whatever weights it
learns, and the measurements agree: the linear map at best reaches about half
the error of the trivial "nothing changes" rule and keeps improving only slowly.

So the question is not what nonlinearity to invent, but whether the nonlinearity
already present is enough. Adding [z, z^2] or [z, z^2, z^3] lets the fit express
the cubic term directly.

This is not a search for a good architecture. The candidates are:

    linear   [z]
    square   [z, z^2]
    cubic    [z, z^2, z^3]

scored against the same previous-step baseline, out of sample, at the same
projection width. If the cubic extension does not help, the missing piece is
something else and adding more terms is not the answer.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_l1_poly.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.runtime.process_1_fast import FHNNetwork, L1Predictor


def expand(Z, power):
    """Feature map: [z, z^2, ...] up to the given power."""
    parts = [Z]
    cur = Z.copy()
    for _ in range(2, power + 1):
        cur = cur * Z
        parts.append(cur)
    return np.concatenate(parts, axis=1)


def score(Z, power, history=1):
    """Ridge fit on the first half, scored on the second, out of sample."""
    if history > 1:
        n = Z.shape[0]
        rows = n - history
        base = np.empty((rows, Z.shape[1] * history))
        for i in range(history):
            # lag i+1 back: rows align with targets Z[history:]
            start = history - 1 - i
            base[:, i * Z.shape[1]:(i + 1) * Z.shape[1]] = Z[start:start + rows]
        target = Z[history:]
    else:
        base = Z[:-1]
        target = Z[1:]

    F = expand(base, power)
    n = F.shape[0]
    h = n // 2
    Ftr, Ytr = F[:h], target[:h]
    Fte, Yte = F[h:], target[h:]

    lam = 1e-3 * max(float(np.trace(Ftr.T @ Ftr)) / Ftr.shape[1], 1e-12)
    W = np.linalg.solve(Ftr.T @ Ftr + lam * np.eye(Ftr.shape[1]), Ftr.T @ Ytr)
    err = np.linalg.norm(Yte - Fte @ W, axis=1).mean()

    # baseline: predict the last step
    base_err = np.linalg.norm(Yte - base[h:, :Z.shape[1]], axis=1).mean()
    return float(err), float(base_err), F.shape[1]


def main():
    net = FHNNetwork(seed=42)
    vecs = []
    for _ in range(16000):
        vecs.append(net.step()["E_rate_vec"])
    Xraw = np.asarray(vecs, dtype=float)

    p = L1Predictor(input_dim=Xraw.shape[1])
    b = p.set_baseline(Xraw[:4000])
    Z = Xraw @ p.proj
    print(f"projection dim {p.proj_dim}, {b['variance_explained']:.1%} variance")
    print()

    print(f"{'form':>8} {'hist':>5} {'features':>9} {'err':>9} {'prev':>9} {'ratio':>7}")
    rows = []
    for power in (1, 2, 3):
        for hist in (1, 2, 4):
            err, base, nf = score(Z, power, hist)
            r = err / base if base > 0 else float("inf")
            name = {1: "linear", 2: "square", 3: "cubic"}[power]
            rows.append((name, hist, nf, err, base, r))
            print(f"{name:>8} {hist:>5} {nf:>9} {err:>9.4f} {base:>9.4f} {r:>7.3f}")

    print()
    print("=" * 58)
    best = min(rows, key=lambda t: t[5])
    linear_best = min([r for r in rows if r[0] == "linear"], key=lambda t: t[5])
    print(f"best overall : {best[0]} hist={best[1]} ratio={best[5]:.3f} "
          f"({best[2]} features)")
    print(f"best linear  : hist={linear_best[1]} ratio={linear_best[5]:.3f}")
    if best[5] < linear_best[5] * 0.9 and best[0] != "linear":
        print("VERDICT: the nonlinear terms help substantially. The cubic term "
              "in the oscillator's equation is what the linear map could not "
              "represent.")
    elif best[0] == "linear":
        print("VERDICT: linear is already the best; the nonlinearity is not the "
              "bottleneck.")
    else:
        print("VERDICT: nonlinear helps a little but not much. Something else "
              "is limiting the prediction.")


if __name__ == "__main__":
    main()
