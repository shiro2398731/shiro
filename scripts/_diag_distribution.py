"""
scripts/_diag_distribution.py

Is the depth distribution structured, or just noise with a long tail?

Two checks:
  1. log-log slope (power-law-like behaviour in the tail)
  2. how many distinct depth levels exist above the noise floor

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_distribution.py
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

for _ in range(20000):
    s.landscape.set_noise_amplitude(0.015)
    s.step()

sp = s.state_space
attrs = sp.attractors()
amps = np.sort(np.array([a.amp for a in attrs]))[::-1]

print(f"attractors: {len(amps)}")
print(f"max {amps.max():.4f}  min {amps.min():.6f}  sum {amps.sum():.2f}")
print()

# rank-size on log-log
ranks = np.arange(1, len(amps) + 1, dtype=float)
mask = amps > 1e-4
x = np.log(ranks[mask])
y = np.log(amps[mask])
if len(x) > 10:
    # fit only the middle range, where power-law behaviour would show
    lo, hi = len(x) // 10, len(x) * 9 // 10
    slope, intercept = np.polyfit(x[lo:hi], y[lo:hi], 1)
    print(f"rank-size log-log slope (middle 80%): {slope:.3f}")
    print("  slope near -1 : Zipf-like, one dominant scale")
    print("  slope near 0  : everything equal, no structure")
    print("  slope << -1   : one region dominates heavily")

# how much of the total depth sits in the top k
total = amps.sum()
print()
for k in (1, 5, 10, 20, 50, 100):
    if k <= len(amps):
        print(f"  top {k:3d} hold {amps[:k].sum() / total:.1%} of total depth")

print()
print("depth bands")
for lo, hi in [(0, 0.005), (0.005, 0.02), (0.02, 0.08), (0.08, 0.3),
               (0.3, 0.7), (0.7, 1.01)]:
    sel = (amps >= lo) & (amps < hi)
    print(f"  {lo:.3f}-{hi:.3f}: {int(sel.sum()):4d}")
