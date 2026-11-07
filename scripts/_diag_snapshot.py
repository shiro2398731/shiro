"""
scripts/_diag_snapshot.py

After a long replay, what does the resulting structure look like in detail?

Answers: is the large jump in salient count a genuine diversification, or one
attractor absorbing its neighbourhood?

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_snapshot.py
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

for _ in range(40000):
    s.landscape.set_noise_amplitude(0.015)
    s.step()

sp = s.state_space
attrs = sp.attractors()
amps = np.array([a.amp for a in attrs])
nm = np.array([len(a.members) for a in attrs])
sig = np.array([a.sigma for a in attrs])

print(f"attractors : {len(attrs)}")
print(f"salient    : {len(sp.salient())}")
print(f"scale      : {sp.global_scale:.6f}")
print()

order = np.argsort(-amps)
print("top 12 by depth")
print("     idx      amp  members     sigma   scale")
for i in order[:12]:
    a = attrs[i]
    print(f"{i:8d} {a.amp:8.4f} {len(a.members):8d} {a.sigma:9.5f} "
          f"{(a.scale or 0):8.5f}")

print()
print("depth buckets")
for lo, hi in [(0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.5),
               (0.5, 0.9), (0.9, 1.01)]:
    sel = (amps >= lo) & (amps < hi)
    k = int(sel.sum())
    if k:
        print(f"  amp {lo:.2f}-{hi:.2f}: {k:4d} regions, "
              f"mean members {nm[sel].mean():.0f}")

print()
print("how much depth is concentrated in the single deepest region")
print(f"  deepest amp            : {amps.max():.4f}")
print(f"  sum of all depths      : {amps.sum():.2f}")
print(f"  share of deepest alone : {amps.max() / amps.sum():.1%}")

print()
print("sigma distribution")
for p in (10, 50, 90, 99):
    print(f"  p{p}: {np.percentile(sig, p):.6f}")
