"""
scripts/_diag_top.py

Inspect the deepest attractors in the real checkpoint.

Answers: what do the deepest regions actually look like, and where is the
current state relative to them?

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_top.py
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
attrs = sp.attractors()
amps = np.array([a.amp for a in attrs])
order = np.argsort(-amps)

print(f"attractors: {len(attrs)}   scale now: {sp.global_scale:.6f}")
print()
print("top 8 by depth")
print("  idx      amp  members      sigma      scale   dist  in-sigma")
for i in order[:8]:
    a = attrs[i]
    d = float(np.linalg.norm(s._st.x - a.mu))
    z = d / a.sigma
    print(f"{i:5d} {a.amp:8.4f} {len(a.members):8d} {a.sigma:10.5f} "
          f"{(a.scale or 0):10.5f} {d:7.3f} {z:9.2f}")

print()
print("depth distribution")
for p in (50, 90, 95, 99, 100):
    print(f"  p{p}: {np.percentile(amps, p):.4f}")

nm = np.array([len(a.members) for a in attrs])
print()
print(f"attractors with >16 members: {int((nm > 16).sum())}")
print(f"attractors with ==64 members (buffer full): {int((nm == 64).sum())}")
