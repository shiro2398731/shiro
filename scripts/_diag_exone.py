"""
scripts/_diag_exone.py

What does the structure look like if the single dominant region is excluded?

If one region holds 94% of all assignments, every aggregate statistic is
describing that region rather than the structure. This reports the same
statistics with and without it.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_exone.py
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

attrs = sp.attractors()
assigned = np.array([a._assigned for a in attrs], dtype=float)
amps = np.array([a.amp for a in attrs], dtype=float)
sig = np.array([a.sigma for a in attrs], dtype=float)
nm = np.array([len(a.members) for a in attrs], dtype=float)

top = int(np.argmax(assigned))
keep = np.ones(len(attrs), dtype=bool)
keep[top] = False

print(f"dominant region: index {top}")
print(f"  assigned {assigned[top]:.0f} ({assigned[top] / assigned.sum():.1%})")
print(f"  amp {amps[top]:.4f}  sigma {sig[top]:.6f}")
print()

for label, sel in (("all", np.ones(len(attrs), bool)), ("without it", keep)):
    a_ = amps[sel]
    s_ = sig[sel]
    n_ = nm[sel]
    print(f"--- {label} ({int(sel.sum())} regions) ---")
    print(f"  amp   p50 {np.median(a_):.5f}  p90 {np.percentile(a_, 90):.5f}  "
          f"max {a_.max():.5f}")
    print(f"  sigma p50 {np.median(s_):.6f}  p90 {np.percentile(s_, 90):.6f}")
    print(f"  members p50 {np.median(n_):.0f}  p90 {np.percentile(n_, 90):.0f}")
    print(f"  regions with >=32 members: {int((n_ >= 32).sum())}")
    print()

# how many regions would still be salient without it
sal_all = sp.salient()
sal_without = [a for a in sal_all if a is not attrs[top]]
print(f"salient now            : {len(sal_all)}")
print(f"salient without it     : {len(sal_without)}")
