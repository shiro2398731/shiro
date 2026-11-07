"""
scripts/_diag_capacity.py

How many attractors does the space actually need?

Runs with the capacity cap raised far above normal so the natural count can be
observed, and reports how it settles. Nothing is written.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_capacity.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream
import src.common.persist as persist

s = Stream(dim=128, max_attractors=100000)
if not persist.load_l5(s):
    print("no checkpoint")
    sys.exit(1)
s.state_space.max_attractors = 100000

marks = {1000, 5000, 10000, 20000, 40000}

for i in range(1, 40001):
    s.landscape.set_noise_amplitude(0.015)
    s.step()
    if i in marks:
        sp = s.state_space
        attrs = sp.attractors()
        amps = np.array([a.amp for a in attrs])
        sig = np.array([a.sigma for a in attrs])
        nm = np.array([len(a.members) for a in attrs])
        print(f"step {i:6d}: att={len(attrs):5d}  "
              f"att>=0.05={int((amps >= 0.05).sum()):4d}  "
              f"salient={len(sp.salient()):3d}  "
              f"sigma_p50={np.median(sig):.4f}  "
              f"scale={sp.global_scale:.4f}  "
              f"fullbuf={int((nm >= 32).sum()) / len(nm):.1%}")
