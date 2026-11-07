"""
scripts/_diag_offstructure.py

How much of the trajectory is spent outside any region?

If a large fraction of states are not assigned to any attractor, the space is
being traversed without leaving structure, and no amount of regulation on
existing regions can change that.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_offstructure.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream
import src.common.persist as persist

s = Stream(dim=128)
if not persist.load_l5(s):
    raise SystemExit("no checkpoint")
s._st.v = np.zeros(128)
sp = s.state_space

# measure assignment hit rate over a run
hits = 0
misses = 0
for _ in range(1500):
    before = sp._total_steps
    s.landscape.set_noise_amplitude(0.015)
    a = sp.record  # not used; step() calls record internally
    s.step()

attrs = sp.attractors()
assigned = np.array([a._assigned for a in attrs], dtype=float)
total = max(sp._total_steps, 1)

print(f"attractors               : {len(attrs)}")
print(f"total observations       : {total}")
print(f"total assignments        : {assigned.sum():.0f}")
print(f"assignment coverage      : {assigned.sum() / total:.1%}")
print()
print("so the fraction of observations that fell inside some region is the")
print("coverage above; the rest were in space with no structure")
print()
print("regions by assignment count")
for lo, hi in [(1, 1), (2, 3), (4, 15), (16, 63), (64, 10**9)]:
    sel = (assigned >= lo) & (assigned <= hi)
    k = int(sel.sum())
    if k:
        print(f"  {lo:>3}-{hi if hi < 10**9 else '+':<6}: {k:4d} regions, "
              f"{assigned[sel].sum() / max(assigned.sum(), 1):.1%} of all assignments")

print()
print("how many regions carry most of the visits")
order = np.argsort(-assigned)
for k in (1, 5, 10, 20, 40):
    if k <= len(order):
        share = assigned[order[:k]].sum() / max(assigned.sum(), 1)
        print(f"  top {k:3d} regions hold {share:.1%} of all assignments")
