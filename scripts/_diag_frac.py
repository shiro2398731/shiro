"""
scripts/_diag_frac.py

Why is most of the structure below its own carrying level?

The carrying level is the depth a region can hold given how often it is
reached. If most regions sit below it, either:

  a) the level is computed wrongly, or
  b) the structure is genuinely thinner than its visit rates would allow,
     which would mean depth is being lost for a reason other than the
     modelled recovery.

This distinguishes them by checking the depth balance directly:
for each region, what does one step actually add and remove?

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_frac.py
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

for _ in range(600):
    s.landscape.set_noise_amplitude(0.015)
    s.step()

sp = s.state_space
attrs = sp.attractors()
n = len(attrs)
print(f"attractors: {n}")
print(f"total_steps observed: {sp._total_steps}")
print()

assigned = np.array([a._assigned for a in attrs], dtype=float)
amps = np.array([a.amp for a in attrs], dtype=float)
sig = np.array([a.sigma for a in attrs], dtype=float)

print("assigned count distribution")
for p in (10, 50, 90, 99):
    print(f"  p{p}: {np.percentile(assigned, p):.0f}")
print(f"  max: {assigned.max():.0f}")
print()

# the per-step balance for each region
a_ratio = sp.assign_ratio
rho = sp.recover_rate
every = max(sp.recover_every, 1)

gain_per_step = (assigned / max(sp._total_steps, 1)) * a_ratio * (1.0 - amps)
loss_per_step = amps * rho / every
net = gain_per_step - loss_per_step

print("per-step balance")
print(f"  net > 0 (gaining): {int((net > 0).sum())} / {n}")
print(f"  net < 0 (losing) : {int((net < 0).sum())} / {n}")
print(f"  median net       : {np.median(net):.3e}")
print()

print("if net is negative for most regions, depth should be falling globally")
print(f"  median amp now   : {np.median(amps):.6f}")
