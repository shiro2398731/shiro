"""
scripts/_test_rest.py

Does the rest field let the trajectory keep moving with no noise at all?

This is the question that decides whether L5 is a system with its own source of
motion or a settling device that needs to be pushed. The run here sets the
fluctuation amplitude to exactly zero for the whole duration, so any motion
that remains comes from the structure itself.

Reported per checkpoint:
    att     attractor count
    edges   L3 edges
    speed   |v|
    trans   cumulative transitions
    rest    mean rest across regions

Reading it:
    speed stays above zero throughout    the structure carries its own motion
    speed decays toward zero             it still settles; rest is too weak
    speed oscillates                     regions take turns, which is the
                                         intended behaviour
    rest saturates at 1 everywhere       fatigue outruns recovery; the whole
                                         landscape is being erased

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_rest.py --steps 20000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import src.common.persist as persist
from src.L5_cognition.stream import Stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--every", type=int, default=2000)
    ap.add_argument("--from-archive", action="store_true")
    args = ap.parse_args()

    s = Stream(dim=128)
    if args.from_archive:
        print(f"archive loaded: {persist.load_l5(s)}")

    try:
        s.step()
    except RuntimeError:
        print("not calibrated; using a synthetic calibration")
        s.calibrate(np.random.default_rng(1).normal(0.5, 0.1, (64, 128)))

    transitions = 0
    rows = []
    for i in range(1, args.steps + 1):
        # No noise at all. Any motion has to come from the rest field.
        s.landscape.set_noise_amplitude(0.0)
        ev = s.step()
        if ev is not None:
            transitions += 1
        if i % args.every == 0 or i == 1:
            ss = s.stream_state()
            r = s.rest.summary()
            rows.append((i, ss["n_attractors"], ss["n_edges"],
                         ss["speed"], transitions, r["u_mean"], r["u_max"]))

    print()
    print(f"{'step':>6} {'att':>5} {'edges':>6} {'speed':>8} {'trans':>6} "
          f"{'rest':>7} {'restmax':>8}")
    for r in rows:
        print(f"{r[0]:>6} {r[1]:>5} {r[2]:>6} {r[3]:>8.4f} {r[4]:>6} "
              f"{r[5]:>7.4f} {r[6]:>8.4f}")

    print()
    speeds = np.array([r[3] for r in rows[1:]])
    rests = np.array([r[5] for r in rows[1:]])
    print("=" * 50)
    if speeds.size == 0:
        print("no checkpoints")
        return
    print(f"speed  min={speeds.min():.4f} max={speeds.max():.4f} "
          f"mean={speeds.mean():.4f}")
    print(f"rest   min={rests.min():.4f} max={rests.max():.4f}")
    print(f"transitions total: {transitions}")

    if speeds[-3:].mean() < 0.01 * max(speeds.max(), 1e-9) + 1e-6:
        print("VERDICT: the state still settles. Rest is not keeping it moving.")
    elif rests.max() > 0.95:
        print("VERDICT: rest saturates. The landscape is being erased rather "
              "than made mobile.")
    elif speeds.std() > 0.2 * max(speeds.mean(), 1e-9):
        print("VERDICT: the state moves without noise and the speed varies. "
              "Regions are taking turns.")
    else:
        print("VERDICT: the state moves without noise, at a roughly steady "
              "rate.")


if __name__ == "__main__":
    main()
