"""
scripts/_test_l3_balance.py

Where does the edge count settle, if anywhere?

L3 grows an edge every time the state moves directly from one region to another
that it has not visited from that region before, and decays all edges on a
slower clock. Whether the graph converges depends on the ratio of the two rates.

The decay is applied once every `recover_every` steps at rate `weaken_rate`, so
the effective per-step decay is weaken_rate / recover_every. An edge whose
weight starts near 1 needs roughly ln(1/eps) / decay steps to reach eps.

This script does not change anything. It measures the ratio and reports whether
the measured edge count is consistent with a balance point or still climbing.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_l3_balance.py --steps 40000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import src.common.persist as persist
from src.L5_cognition.stream import Stream
from src.L5_cognition.noise import ErrorNoise
from src.runtime.process_1_fast import FHNNetwork, L1Predictor


class Upstream:
    def __init__(self, seed=42):
        self.net = FHNNetwork(seed=seed)
        self.pred = L1Predictor(input_dim=100, proj_dim=8)
        self.noise = ErrorNoise(dim=128)

    def step(self):
        snap = self.net.step()
        self.pred.step(snap["E_rate_vec"])
        return self.noise.observe(self.pred.last_error, self.pred.proj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--every", type=int, default=4000)
    ap.add_argument("--from-archive", action="store_true")
    args = ap.parse_args()

    s = Stream(dim=128)
    if args.from_archive:
        print(f"archive loaded: {persist.load_l5(s)}")
    try:
        s.step()
    except RuntimeError:
        s.calibrate(np.random.default_rng(1).normal(0.5, 0.1, (64, 128)))

    up = Upstream()
    transitions = 0
    rows = []

    for i in range(1, args.steps + 1):
        s.landscape.set_noise_amplitude(up.step())
        ev = s.step()
        if ev is not None:
            transitions += 1
        if i % args.every == 0 or i == 1:
            ss = s.stream_state()
            g = s.graph.stats()
            rows.append((i, ss["n_attractors"], g["edges"], transitions,
                         g["total_w"], g["mean_w"]))

    print()
    print(f"{'step':>7} {'att':>5} {'edges':>6} {'trans':>7} {'totW':>10} "
          f"{'meanW':>8} {'e/att':>7}")
    prev = None
    deltas = []
    for r in rows:
        ratio = r[2] / max(r[1], 1)
        print(f"{r[0]:>7} {r[1]:>5} {r[2]:>6} {r[3]:>7} {r[4]:>10.2f} "
              f"{r[5]:>8.4f} {ratio:>7.3f}")
        if prev is not None:
            deltas.append(r[2] - prev)
        prev = r[2]

    print()
    print("=" * 55)
    if deltas:
        print(f"edge gain per checkpoint: {deltas}")
        print(f"last gain: {deltas[-1]} (trend: "
              f"{'shrinking' if len(deltas) > 1 and deltas[-1] < deltas[0] else 'growing'})")

    # The balance point implied by the two rates, for comparison with what was
    # measured. Not a prediction -- just the arithmetic of the two constants.
    decay_per_step = s.consolidator.weaken_rate / max(s._recover_every, 1)
    print(f"decay per step: {decay_per_step:.6f}")
    if decay_per_step > 0:
        life = np.log(1.0 / 0.01) / decay_per_step
        print(f"steps for an edge to fall from 1.0 to 0.01: {life:.0f}")
    rate = (rows[-1][3] - rows[0][3]) / max(rows[-1][0] - rows[0][0], 1)
    print(f"transitions per step: {rate:.5f}")
    if decay_per_step > 0 and rate > 0:
        print(f"implied steady-state edge count: "
              f"{rate * np.log(1.0 / 0.01) / decay_per_step:.0f}")


if __name__ == "__main__":
    main()
