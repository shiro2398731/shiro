"""
scripts/_diag_final.py

Final check of the current L4 configuration over a longer run.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_final.py
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

print(f"beta = {s.regulation.beta}   wall_enabled = {s._wall_enabled}")

for i in range(3000):
    s.landscape.set_noise_amplitude(0.015)
    s.step()
    if i in (499, 1499, 2999):
        nm = np.array([len(a.members) for a in sp.attractors()])
        v = s.regulation.summary(sp)
        print(f"step {i + 1:4d}: att {len(sp):4d}  "
              f"rev32 {int((nm >= 32).sum()):3d}  "
              f"salient {len(sp.salient()):2d}  "
              f"speed {float(np.linalg.norm(s._st.v)):6.3f}  "
              f"frac_below {v['frac_below']:.3f}")
