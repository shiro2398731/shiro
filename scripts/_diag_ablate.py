"""
scripts/_diag_ablate.py

Ablation: what does each part of the L4 machinery actually do?

Conditions:
    none   -- regulation off, wall off
    pull   -- regulation on, wall off
    wall   -- regulation off, wall on
    both   -- both on

The point is to attribute effects, not to find the best-sounding number.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_ablate.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream
import src.common.persist as persist

STEPS = 800


def run(pull: bool, wall: bool, wall_gain: float = 1.0):
    s = Stream(dim=128)
    if not persist.load_l5(s):
        raise SystemExit("no checkpoint")
    s._st.v = np.zeros(128)
    sp = s.state_space
    s.regulation.beta = 0.5 if pull else 0.0
    sp.set_wall(None)

    x0 = s._st.x.copy()
    dist = []

    for i in range(STEPS):
        s.landscape.set_noise_amplitude(0.015)
        s.step()
        if i % 16 == 0:
            sp.set_wall(s.regulation.viability.wall(sp) if wall else None,
                        gain=wall_gain)
        dist.append(float(np.linalg.norm(s._st.x - x0)))

    attrs = sp.attractors()
    nm = np.array([len(a.members) for a in attrs])
    R = max(sp.global_scale, 1e-12)
    regions = s.regulation.viability.measure(sp)
    return {
        "attractors": len(attrs),
        "rev16": int((nm >= 16).sum()),
        "rev32": int((nm >= 32).sum()),
        "salient": len(sp.salient()),
        "speed": float(np.linalg.norm(s._st.v)),
        "travel_R": float(max(dist)) / R,
        "frac_below": float(np.mean([r.ratio < 1.0 for r in regions])),
    }


conds = [
    ("none", run(False, False)),
    ("pull", run(True, False)),
    ("wall", run(False, True)),
    ("both", run(True, True)),
]

keys = ["attractors", "rev16", "rev32", "salient", "speed", "travel_R",
        "frac_below"]
print(f"{'':<10}" + "".join(f"{c[0]:>12}" for c in conds))
for k in keys:
    cells = []
    for _, d in conds:
        v = d[k]
        cells.append(f"{v:>12.4f}" if isinstance(v, float) else f"{v:>12}")
    print(f"{k:<10}" + "".join(cells))
