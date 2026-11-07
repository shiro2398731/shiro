"""
scripts/_test_l4_ablation.py

What does the drive layer actually do to the trajectory?

Drive computes a tilt from unresolved tensions and adds it to the potential's
slope. That is a mechanism; whether it changes anything is a separate question.
This runs the same system twice from the same archive with the same upstream,
once with the tilt and once without, and compares.

What is compared:
    att        attractor count reached
    edges      L3 edges reached
    speed      mean state speed
    trans      transitions
    spread     how widely the trajectory moved (mean distance from the start)

If the two runs are indistinguishable, the drive layer is not doing anything and
the tension computation is decoration. If they diverge, the comparison says how
much it matters.

Both runs share the seed, so the difference is the drive term and nothing else.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_l4_ablation.py --steps 12000
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
    """The surprise series, identical in both runs because the seed is fixed."""

    def __init__(self, seed=42):
        self.net = FHNNetwork(seed=seed)
        self.pred = L1Predictor(input_dim=100, proj_dim=8)
        self.noise = ErrorNoise(dim=128)

    def step(self):
        snap = self.net.step()
        self.pred.step(snap["E_rate_vec"])
        return self.noise.observe(self.pred.last_error, self.pred.proj)


class SilentDrive:
    """Replaces Drive with something that produces no tilt, changing nothing else."""

    def __init__(self, dim=128):
        self.D = dim
        self._last = []

    def deflection(self, state_space, graph):
        return None

    def set_baseline(self, grad_norms):
        return {"ok": False, "reason": "silent drive"}

    def summary(self):
        return {"n": 0, "sum": 0.0}


def run(with_drive: bool, steps: int):
    s = Stream(dim=128)
    persist.load_l5(s)
    try:
        s.step()
    except RuntimeError:
        s.calibrate(np.random.default_rng(1).normal(0.5, 0.1, (64, 128)))

    if not with_drive:
        s.drive = SilentDrive(dim=128)

    up = Upstream()
    start = s._st.x.copy()
    speeds = []
    trans = 0
    for _ in range(steps):
        s.landscape.set_noise_amplitude(up.step())
        ev = s.step()
        if ev is not None:
            trans += 1
        speeds.append(s.stream_state()["speed"])

    ss = s.stream_state()
    g = s.graph.stats()
    return {
        "att": ss["n_attractors"],
        "edges": g["edges"],
        "speed": float(np.mean(speeds)),
        "speed_std": float(np.std(speeds)),
        "trans": trans,
        "spread": float(np.linalg.norm(s._st.x - start)),
        "tensions": len(s.drive.last) if hasattr(s.drive, "last") else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    args = ap.parse_args()

    print(f"running {args.steps} steps with drive ...")
    a = run(True, args.steps)
    print(f"running {args.steps} steps without drive ...")
    b = run(False, args.steps)

    print()
    print(f"{'metric':>10} {'with':>10} {'without':>10} {'change':>10}")
    for k in ("att", "edges", "speed", "speed_std", "trans", "spread", "tensions"):
        va, vb = a[k], b[k]
        if isinstance(va, int):
            print(f"{k:>10} {va:>10} {vb:>10} {va - vb:>10}")
        else:
            rel = (va - vb) / max(abs(vb), 1e-12)
            print(f"{k:>10} {va:>10.4f} {vb:>10.4f} {rel:>9.1%}")

    print()
    print("=" * 50)
    same = (abs(a["speed"] - b["speed"]) / max(abs(b["speed"]), 1e-12) < 0.02
            and abs(a["edges"] - b["edges"]) / max(b["edges"], 1) < 0.05)
    if same:
        print("VERDICT: the drive layer does not change the trajectory. The "
              "tension computation is not affecting anything.")
    else:
        print("VERDICT: the drive layer changes the trajectory.")


if __name__ == "__main__":
    main()
