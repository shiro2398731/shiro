"""
scripts/_diag_converge.py

Does the attractor count stabilise, or does it keep hitting the cap?

Replays the real checkpoint forward for 20000 steps and reports the structure
at intervals. Writes nothing.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_converge.py
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.landscape import Landscape
from src.L5_cognition.dynamics import Dynamics, DynamicState

PATH = os.path.join("data", "state", "l5.pkl")
with open(PATH, "rb") as f:
    obj = pickle.load(f)

zero = np.asarray(obj["zero"], dtype=float)
x0 = np.asarray(obj["x"], dtype=float)
v0 = np.asarray(obj["v"], dtype=float)

land = Landscape(dim=128)
land._zero = zero.copy()
land.set_noise_amplitude(0.015)
dyn = Dynamics(dim=128, gamma=0.05, dt=0.01)
st = DynamicState(x=x0.copy(), v=v0.copy())
rng = np.random.default_rng(11)

marks = {3000, 6000, 10000, 15000, 20000}

for i in range(1, 20001):
    st = dyn.step(st, land, rng)
    land.record(st.x)
    if i % 16 == 0:
        land.recover()
    if i in marks:
        ss = land.state_space
        amps = np.array([a.amp for a in ss.attractors()])
        shares = np.array([len(a.members) for a in ss.attractors()], dtype=float)
        print(f"step {i:6d}: attractors={len(amps):4d}  "
              f"salient={len(ss.salient()):3d}  "
              f"amp_max={amps.max():.3f}  amp_p50={np.median(amps):.3f}  "
              f"max_share={shares.max() / shares.sum():.1%}  "
              f"gscale={ss.global_scale:.4f}")

# member distribution at the end
ss = land.state_space
nm = np.array([len(a.members) for a in ss.attractors()], dtype=float)
print("\nfinal member counts:")
print(f"  median {np.median(nm):.0f}   p90 {np.percentile(nm, 90):.0f}   "
      f"max {nm.max():.0f}")
print(f"  singleton regions: {int((nm == 1).sum())} / {len(nm)}")
