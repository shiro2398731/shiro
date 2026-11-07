"""
scripts/_diag_live.py

Replay the real checkpoint forward with the current code and print the
structure at intervals. Writes nothing.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_live.py
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

print(f"loaded: attractors={len(s.state_space)} "
      f"gscale={s.state_space.global_scale:.6f}")

marks = {100, 500, 1000, 2000, 4000}

for i in range(1, 4001):
    s.landscape.set_noise_amplitude(0.015)
    s.step()
    if i in marks:
        ss = s.stream_state()
        amps = np.array([a.amp for a in s.state_space.attractors()])
        print(f"step {i:5d}: gscale={ss['global_scale']:.5f}  "
              f"att={ss['n_attractors']:4d}  salient={ss['n_salient']:3d}  "
              f"edges={ss['n_edges']:3d}  amp>0.1={int((amps > 0.1).sum()):3d}  "
              f"tensions={ss['tensions']['n']}")
