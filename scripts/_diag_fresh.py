"""
scripts/_diag_fresh.py

Cold start: no checkpoint. Does the structure form sanely from nothing?

This matters because after the changes the scale is derived from existing
structure, and at the very start there is none.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_fresh.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream

s = Stream(dim=128, seed=7)

print("=== cold start ===")
print(f"scale before calibration : {s.state_space.global_scale:.6f}")

warm = np.cumsum(np.random.default_rng(1).normal(0, 0.05, (64, 128)), axis=0)
s.calibrate(warm)

rng = np.random.default_rng(99)
drift = np.zeros(128)

marks = {20, 100, 500, 2000, 8000}

for i in range(1, 8001):
    drift += rng.normal(0, 0.02, 128) * 0.3
    s.landscape.set_noise_amplitude(0.02)
    s.landscape.set_deflection(drift)
    s.step()
    if i in marks:
        ss = s.stream_state()
        attrs = s.state_space.attractors()
        amps = np.array([a.amp for a in attrs]) if attrs else np.zeros(1)
        print(f"step {i:5d}: att={ss['n_attractors']:4d}  "
              f"att>0.1={int((amps > 0.1).sum()):3d}  "
              f"salient={ss['n_salient']:3d}  edges={ss['n_edges']:4d}  "
              f"tensions={ss['tensions']['n']:3d}  "
              f"gscale={ss['global_scale']:.4f}")

print("\n=== failure modes to watch for ===")
print("  gscale stuck at 1.0        : the scale never seeds")
print("  gscale oscillating wildly  : structure-based measure unstable")
print("  att growing without bound  : fragmentation returned")
print("  edges == 0                 : no transitions, L3 starved")
print("  NaN in state               : integrator diverged")
print()
print(f"  x finite : {bool(np.all(np.isfinite(s._st.x)))}")
print(f"  v norm   : {np.linalg.norm(s._st.v):.4f}")
