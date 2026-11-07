"""
scripts/_diag_scale_fix.py

Check whether the new global-scale update behaves correctly in the two
situations that broke the old one:

  1. oscillation in place (trapped): the scale must not collapse
  2. genuine travel: the scale must follow the observed range

Also replays a short window of the real trajectory and reports how many
attractors form and how concentrated they are.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_scale_fix.py
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.terrain import StateSpace

print("=== case 1: oscillation in place ===")
sp = StateSpace(dim=128)
base = np.zeros(128)
for i in range(400):
    x = base + 0.002 * np.sin(np.arange(128) * 0.1 + i * 0.05)
    sp._update_global_scale(x)
print(f"  global_scale after 400 near-static steps: {sp.global_scale:.6f}")
print(f"  (old behaviour would have collapsed to ~1e-4)")

print("\n=== case 2: genuine travel ===")
sp2 = StateSpace(dim=128)
rng = np.random.default_rng(3)
x = np.zeros(128)
for i in range(400):
    x = x + rng.normal(0, 0.05, 128)
    sp2._update_global_scale(x)
mad = float(np.linalg.norm(0.05 * rng.normal(0, 1, 128)))
print(f"  global_scale after 400 travelling steps : {sp2.global_scale:.6f}")
print(f"  step magnitude ~ {mad:.6f}")

print("\n=== case 3: travel then stop ===")
sp3 = StateSpace(dim=128)
x = np.zeros(128)
for i in range(300):
    x = x + rng.normal(0, 0.05, 128)
    sp3._update_global_scale(x)
before = sp3.global_scale
for i in range(400):
    sp3._update_global_scale(x + 0.002 * np.sin(np.arange(128) * 0.1 + i * 0.05))
after = sp3.global_scale
print(f"  after travel : {before:.6f}")
print(f"  after 400 static steps : {after:.6f}   ratio {after/before:.3f}")

# ---------- replay real window ----------

print("\n=== replay on real checkpoint ===")
PATH = os.path.join("data", "state", "l5.pkl")
with open(PATH, "rb") as f:
    obj = pickle.load(f)

zero = np.asarray(obj["zero"], dtype=float)
x0 = np.asarray(obj["x"], dtype=float)
v0 = np.asarray(obj["v"], dtype=float)

from src.L5_cognition.landscape import Landscape
from src.L5_cognition.dynamics import Dynamics, DynamicState

land = Landscape(dim=128)
land._zero = zero.copy()
land.set_noise_amplitude(0.015)
dyn = Dynamics(dim=128, gamma=0.05, dt=0.01)
st = DynamicState(x=x0.copy(), v=v0.copy())
rng2 = np.random.default_rng(11)

for _ in range(3000):
    st = dyn.step(st, land, rng2)
    land.record(st.x)

ss = land.state_space
counts = np.array([len(a.members) for a in ss.attractors()], dtype=float)
amps = np.array([a.amp for a in ss.attractors()], dtype=float)
sig = np.array([a.sigma for a in ss.attractors()], dtype=float)

print(f"  attractors      : {len(counts)}")
print(f"  global_scale    : {ss.global_scale:.6f}")
print(f"  largest share   : {counts.max()/counts.sum():.1%}" if len(counts) else "")
print(f"  sigma range     : {sig.min():.6f} .. {sig.max():.6f}" if len(sig) else "")
print(f"  amp max         : {amps.max():.5f}" if len(amps) else "")
print(f"  amp > 0.1       : {int((amps > 0.1).sum())}")
