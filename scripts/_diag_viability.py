"""
scripts/_diag_viability.py

Does regulation change behaviour, and does it change it in the intended
direction?

Checks, on the real checkpoint:
    1. carrying depth matches the analytic prediction
    2. what fraction of regions are below their own carrying level
    3. does regulation increase the amount of revisiting (recurrence)
    4. does it break anything (NaN, runaway, collapse)

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_viability.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream
import src.common.persist as persist


def run(with_regulation: bool, steps: int = 3000):
    s = Stream(dim=128)
    if not persist.load_l5(s):
        raise SystemExit("no checkpoint")

    if not with_regulation:
        s.regulation.beta = 0.0        # ratio**0 == 1, so pull is neutral

    for _ in range(steps):
        s.landscape.set_noise_amplitude(0.015)
        s.step()

    sp = s.state_space
    attrs = sp.attractors()
    amps = np.array([a.amp for a in attrs])
    nm = np.array([len(a.members) for a in attrs])
    return {
        "attractors": len(attrs),
        "revisited16": int((nm >= 16).sum()),
        "revisited32": int((nm >= 32).sum()),
        "salient": len(sp.salient()),
        "edges": len(s.graph),
        "tensions": len(s.drive.last),
        "amp_p50": float(np.median(amps)),
        "amp_p90": float(np.percentile(amps, 90)),
        "speed": float(np.linalg.norm(s._st.v)),
        "finite": bool(np.all(np.isfinite(s._st.x))),
        "viability": s.regulation.viability.summary(sp),
    }


print("=== 1. carrying depth, analytic check ===")
s = Stream(dim=128)
persist.load_l5(s)
sp = s.state_space
from src.L4_drive.viability import Viability
att_ = sp.attractors()[:5]
total = max(sp._total_steps, 1)
for a in att_:
    p = a._assigned / total
    got = Viability.carrying_depth(sp, a, p)
    print(f"  amp {a.amp:.4f}  assigned {a._assigned:>6}  p {p:.6f}  "
          f"carrying {got:.6f}  ratio {a.amp / got if got > 0 else 0:.3f}")

print()
print("=== 2. viability distribution at load ===")
v = s.regulation.viability.summary(sp)
for k, val in v.items():
    print(f"  {k}: {val}")

print()
print("=== 3. regulation off ===")
off = run(False)
for k, val in off.items():
    print(f"  {k}: {val}")

print()
print("=== 4. regulation on (beta 0.5) ===")
on = run(True)
for k, val in on.items():
    print(f"  {k}: {val}")

print()
print("=== comparison ===")
print(f"{'metric':<14} {'off':>10} {'on':>10}")
for k in ("attractors", "revisited16", "revisited32", "salient", "edges",
          "tensions", "amp_p50", "amp_p90", "speed"):
    print(f"{k:<14} {off[k]:>10.4f} {on[k]:>10.4f}"
          if isinstance(off[k], float) else
          f"{k:<14} {off[k]:>10} {on[k]:>10}")
