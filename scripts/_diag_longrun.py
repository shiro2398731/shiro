"""
scripts/_diag_longrun.py

Does the depth distribution settle, or does everything end up equally
shallow?

Runs a long replay and reports the depth distribution at several points,
together with how many regions are actually being revisited.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_longrun.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream
import src.common.persist as persist

s = Stream(dim=128)
if not persist.load_l5(s):
    print("no checkpoint")
    sys.exit(1)

marks = {500, 2000, 5000, 10000, 20000, 40000}
t_start = time.time()

for i in range(1, 40001):
    s.landscape.set_noise_amplitude(0.015)
    s.step()
    if i in marks:
        ss = s.stream_state()
        attrs = s.state_space.attractors()
        amps = np.array([a.amp for a in attrs])
        nm = np.array([len(a.members) for a in attrs])
        # member count is capped at 64; a region that keeps being visited has
        # a full buffer and a depth that reflects sustained assignment
        visit_ratio = int((nm >= 32).sum()) / max(len(nm), 1)
        print(f"step {i:6d}: att={len(amps):4d}  "
              f"amp p50={np.median(amps):.4f}  p90={np.percentile(amps, 90):.4f}  "
              f"p99={np.percentile(amps, 99):.4f}  max={amps.max():.3f}  "
              f"fullbuf={visit_ratio:.1%}  "
              f"salient={ss['n_salient']}  edges={ss['n_edges']}  "
              f"elapsed={time.time() - t_start:.0f}s")
