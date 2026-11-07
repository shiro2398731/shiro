"""
scripts/_test_integrated.py

Run the real L5 stream, with L3 and L4 live inside it, and report what grows.

This is the check that matters: the unit tests confirm each module behaves on
constructed input, but only a running stream shows whether the three of them
together produce something that grows, converges, or runs away.

What is reported per checkpoint:
    att       attractor count
    edges     L3 edges
    asym      mean directional imbalance, 0 = symmetric, 1 = one-way
    tens      L4 tensions
    speed     state speed
    trans     transitions so far

Reading the numbers:
    edges rising steadily        L3 is recording
    edges flat at 0              no transitions are happening at all
    edges rising without bound   L3 is not forgetting; metabolic rate too low
    asym rising                  direction is forming
    tens exploding               the drive loop is feeding itself
    speed near 0                 the state is stuck

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_integrated.py --steps 2000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import src.common.persist as persist
from src.L5_cognition.stream import Stream
from src.L5_cognition.noise import ErrorNoise
from src.runtime.process_1_fast import FHNNetwork, L1Predictor


class Upstream:
    """
    The part of P1 that L5 actually depends on: the surprise series.

    L5's noise amplitude is not a number anyone chooses. It is the current
    surprise measured against the history of surprise -- a fact about how
    predictable the present is compared to the recent past. Standing in for
    that here, with the real L0 and L1, is the difference between testing L5
    and testing an empty potential well.

    An earlier run of this script fed no noise at all, because the landscape
    defaulted the amplitude to 0.0. The state then settled into the nearest
    minimum over 30000 steps and stopped, which looked like a converged system
    and was reported as one.
    """

    def __init__(self, seed=42):
        self.net = FHNNetwork(seed=seed)
        self.pred = L1Predictor(input_dim=100, proj_dim=8)
        self.noise = ErrorNoise(dim=128)

    def step(self):
        """Advance L0 and L1, return the fluctuation vector."""
        snap = self.net.step()
        self.pred.step(snap["E_rate_vec"])
        return self.noise.observe(self.pred.last_error, self.pred.proj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--from-archive", action="store_true",
                    help="load data/state/l5.pkl instead of cold starting")
    ap.add_argument("--every", type=int, default=100)
    args = ap.parse_args()

    s = Stream(dim=128)
    up = Upstream()

    if args.from_archive:
        ok = persist.load_l5(s)
        print(f"archive loaded: {ok}")
        if not ok:
            print("no archive; falling back to cold start")

    # Calibration needs samples of the upstream signal. Cold start has no
    # upstream, so a stationary distribution is used and this is recorded as a
    # limitation rather than hidden.
    try:
        s.step()
    except RuntimeError:
        print("not calibrated; using a synthetic calibration")
        rng = np.random.default_rng(1)
        s.calibrate(rng.normal(0.5, 0.1, (64, 128)))

    t0 = time.time()
    transitions = 0
    last_att = None
    rows = []

    for i in range(1, args.steps + 1):
        # The prediction error, mapped back into the state space, as the
        # fluctuation. Direction and magnitude both come from it.
        amp = up.step()
        s.landscape.set_noise_amplitude(amp)
        ev = s.step()
        if ev is not None:
            transitions += 1
        if i % args.every == 0 or i == 1:
            ss = s.stream_state()
            g = s.graph.stats()
            att = ss["n_attractors"]
            rows.append((
                i, att, g["edges"], g["asymmetry"],
                ss["tensions"]["n"], ss["speed"], transitions,
                float(np.linalg.norm(amp)),
            ))

    print()
    print(f"{'step':>6} {'att':>5} {'edges':>6} {'asym':>6} {'tens':>5} "
          f"{'speed':>7} {'trans':>6} {'noise':>8}")
    for r in rows:
        print(f"{r[0]:>6} {r[1]:>5} {r[2]:>6} {r[3]:>6.3f} {r[4]:>5} "
              f"{r[5]:>7.4f} {r[6]:>6} {r[7]:>8.5f}")

    el = time.time() - t0
    print()
    print(f"{args.steps} steps in {el:.1f}s ({el / max(args.steps,1)*1000:.2f} ms/step)")

    # ---- verdict ----
    print()
    print("=" * 50)
    if len(rows) < 3:
        print("not enough checkpoints to judge")
        return
    e0, e1 = rows[1][2], rows[-1][2]
    t_all = rows[-1][6]
    print(f"transitions during run : {t_all}")
    print(f"edges {e0} -> {e1}")
    if t_all == 0:
        print("VERDICT: no transitions happened; the state never left one attractor.")
        print("         L3 cannot record anything under these conditions.")
    elif e1 == 0:
        print("VERDICT: transitions happened but no edge was recorded. Wiring bug.")
    elif e1 > 0 and e0 == 0:
        print("VERDICT: L3 started recording. Growing.")
    elif e1 > e0:
        print("VERDICT: L3 growing.")
    else:
        print("VERDICT: edges flat or falling; check the metabolic rate.")


if __name__ == "__main__":
    main()
