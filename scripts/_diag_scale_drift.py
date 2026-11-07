"""
scripts/_diag_scale_drift.py

Track how the region scale evolves over a long run, and what drives it.

The scale is used only as the starting width of a newly created attractor.
If it grows without bound, new attractors are born wider and wider, absorb
larger neighbourhoods, and merge away.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_scale_drift.py
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

marks = {500, 1000, 2000, 4000, 8000, 16000, 32000}

print(f"start: att={len(s.state_space)}  scale={s.state_space.global_scale:.4f}")

for i in range(1, 32001):
    s.landscape.set_noise_amplitude(0.015)
    s.step()
    if i in marks:
        sp = s.state_space
        attrs = sp.attractors()
        sig = np.array([a.sigma for a in attrs])
        scales = np.array([a.scale or 0.0 for a in attrs])
        recent = scales[-min(50, len(scales)):]
        print(f"step {i:6d}: att={len(attrs):4d}  "
              f"scale={sp.global_scale:.4f}  "
              f"sigma_p50={np.median(sig):.4f}  "
              f"new50_birth_p50={np.median(recent):.4f}  "
              f"salient={len(sp.salient())}")
