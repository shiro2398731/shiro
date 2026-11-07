"""
scripts/_diag_long.py

Does the structure stay healthy over a longer run, or does it fragment again?

Replays the real checkpoint forward for 20000 steps and reports at intervals.
Writes nothing.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_long.py
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

marks = {500, 2000, 5000, 10000, 15000, 20000}

for i in range(1, 20001):
    s.landscape.set_noise_amplitude(0.015)
    s.step()
    if i in marks:
        ss = s.stream_state()
        amps = np.array([a.amp for a in s.state_space.attractors()])
        print(f"step {i:6d}: att={ss['n_attractors']:4d}  "
              f"att>0.1={int((amps > 0.1).sum()):4d}  "
              f"salient={ss['n_salient']:3d}  edges={ss['n_edges']:4d}  "
              f"tensions={ss['tensions']['n']:3d}  "
              f"amp_p50={np.median(amps):.4f}  gscale={ss['global_scale']:.4f}")
