"""
scripts/_test_l1_baseline.py

Is the predictor better than doing nothing?

A linear predictor can be wrong in a way that is invisible in a log: it always
produces a number, and that number decreases over time as the weights settle, so
"the error is going down" looks like learning whether or not the prediction
carries any information.

The only way to tell is to compare against baselines that use no model at all:

    mean      predict the running mean of the input. The dumbest possible guess.
    previous  predict that the next step equals the last one.
    model     the actual predictor.

If the model does not beat both, the projection and the weights are not
capturing anything about the sequence, and the whole layer is decoration.

The comparison is on the same stream from the same oscillator, so the three are
scored on identical inputs.

The predictor's proj_dim and lr are set from the input's autocorrelation first,
as they are in P1 -- otherwise this would be measuring a configuration that only
exists in the test.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_l1_baseline.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.runtime.process_1_fast import FHNNetwork, L1Predictor


def main():
    net = FHNNetwork(seed=42)
    vecs = []
    for _ in range(6000):
        vecs.append(net.step()["E_rate_vec"])
    X = np.asarray(vecs, dtype=float)

    # Calibrate exactly as P1 does.
    pred = L1Predictor(input_dim=X.shape[1])
    base = pred.set_baseline(X[:4000])
    print(f"calibrated: proj_dim={base['proj_dim']} lr={base['lr']:.4f} "
          f"lambda_max={base['lambda_max']:.3f}")
    print()

    # All three baselines are scored in the same space the model works in.
    # Comparing a 17-dimensional error against a 100-dimensional one is not a
    # comparison; an earlier version of this script did exactly that and
    # reported a 97% improvement that was entirely an artefact of dimension.
    Z = X @ pred.proj

    err_model, err_mean, err_prev = [], [], []
    run_mean = Z[0].copy()

    for i in range(1, Z.shape[0]):
        z = Z[i]

        # model
        pred.step(X[i])
        err_model.append(float(np.linalg.norm(pred.last_error)))

        # mean baseline: predict the running mean of the projection
        err_mean.append(float(np.linalg.norm(z - run_mean)))
        run_mean += (z - run_mean) / (i + 1)

        # previous baseline: predict the last projected step
        err_prev.append(float(np.linalg.norm(z - Z[i - 1])))

    # Score on the second half only: the first half is the predictor's warm-up,
    # and scoring the baselines over a period the model was still learning
    # would flatter the model.
    h = len(err_model) // 2
    m = np.asarray(err_model[h:])
    a = np.asarray(err_mean[h:])
    b = np.asarray(err_prev[h:])

    print(f"{'':>10} {'mean err':>10} {'median':>10}")
    print(f"{'model':>10} {m.mean():>10.4f} {np.median(m):>10.4f}")
    print(f"{'mean':>10} {a.mean():>10.4f} {np.median(a):>10.4f}")
    print(f"{'previous':>10} {b.mean():>10.4f} {np.median(b):>10.4f}")
    print()
    print(f"model vs mean     : {(m.mean() - a.mean()) / a.mean():+.1%}")
    print(f"model vs previous : {(m.mean() - b.mean()) / b.mean():+.1%}")

    print()
    print("=" * 55)
    better_mean = m.mean() < a.mean() * 0.99
    better_prev = m.mean() < b.mean() * 0.99
    if better_mean and better_prev:
        print("VERDICT: the model beats both baselines. The projection and the "
              "weights carry information about the sequence.")
    elif better_mean or better_prev:
        which = "running mean" if better_mean else "previous step"
        print(f"VERDICT: the model beats only the {which}. Partial.")
    else:
        print("VERDICT: the model does not beat either baseline. It is not "
              "predicting anything the trivial guesses do not already get.")


if __name__ == "__main__":
    main()
