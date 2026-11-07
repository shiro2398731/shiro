"""
scripts/_diag_three.py

Measure the three things that were called out:
  1. can she move, or is she stuck
  2. are the attractors real structure, or just shallow dents
  3. is she continuous across a restart

This runs on the real checkpoint. Nothing is written.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_three.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream
import src.common.persist as persist

s = Stream(dim=128)
if not persist.load_l5(s):
    print("no checkpoint")
    sys.exit(1)

sp = s.state_space
print("=== 1. can she move ===")
print(f"attractors : {len(sp)}")
print(f"speed      : {np.linalg.norm(s._st.v):.4f}")
print(f"scale      : {sp.global_scale:.4f}")

# where is she relative to the nearest attractor, in its own units
nearest = sp.nearest_attractor(s._st.x)
if nearest:
    idx, z = nearest
    a = sp.attractors()[idx]
    print(f"nearest    : index {idx}, normalized distance {z:.6f}")
    print(f"  its sigma {a.sigma:.6f}  depth {a.amp:.4f}")

# run and watch how far she actually travels, in units of the region scale
x0 = s._st.x.copy()
R = sp.global_scale
dist = []
for i in range(2000):
    s.landscape.set_noise_amplitude(0.015)
    s.step()
    dist.append(float(np.linalg.norm(s._st.x - x0)))
dist = np.array(dist)
print()
print("travel over 2000 steps, in units of region scale:")
for step in (100, 500, 1000, 2000):
    print(f"  after {step:4d}: {dist[step-1] / R:8.3f} R")
print(f"  furthest     : {dist.max() / R:8.3f} R")

print()
print("=== 2. are the attractors real structure ===")
attrs = sp.attractors()
amps = np.array([a.amp for a in attrs])
nm = np.array([len(a.members) for a in attrs])
sig = np.array([a.sigma for a in attrs])
print(f"depth  p10/p50/p90 : {np.percentile(amps,10):.4f} / "
      f"{np.median(amps):.4f} / {np.percentile(amps,90):.4f}")
print(f"members p10/p50/p90: {np.percentile(nm,10):.0f} / "
      f"{np.median(nm):.0f} / {np.percentile(nm,90):.0f}")
print(f"sigma  p10/p50/p90 : {np.percentile(sig,10):.5f} / "
      f"{np.median(sig):.5f} / {np.percentile(sig,90):.5f}")
print(f"regions revisited (>=16 members): "
      f"{int((nm >= 16).sum())} / {len(nm)}")

# does an attractor's depth predict how much pull it exerts?
# pull at its own edge = amp / (sigma * e^0.5)
pull = amps / (np.maximum(sig, 1e-9) * np.sqrt(np.e))
print(f"pull at edge  p50/max: {np.median(pull):.3f} / {pull.max():.3f}")

print()
print("=== 3. continuity across reload ===")
snap = s.stream_state()
edges_before = snap["n_edges"]
identity_before = s.identity.rounds
x_before = s._st.x.copy()
v_before = s._st.v.copy()

tmp_name = "_diag_three_l5.pkl"
from pathlib import Path
data_dir = Path(os.path.abspath("data"))
persist.save_l5(s, tmp_name)
s2 = Stream(dim=128)
ok = persist.load_l5(s2, tmp_name)

snap2 = s2.stream_state()
print(f"reload ok        : {ok}")
print(f"attractors       : {snap['n_attractors']} -> {snap2['n_attractors']}")
print(f"edges            : {edges_before} -> {snap2['n_edges']}")
print(f"identity rounds  : {identity_before} -> {s2.identity.rounds}")
print(f"step_count       : {s._step_count} -> {s2._step_count}")
print(f"position drift   : {float(np.linalg.norm(s2._st.x - x_before)):.2e}")
print(f"velocity drift   : {float(np.linalg.norm(s2._st.v - v_before)):.2e}")

leftover = data_dir / tmp_name
if leftover.exists():
    leftover.unlink()
