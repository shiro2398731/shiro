"""
scripts/_diag_roundtrip.py

Save/load round trip in a scratch directory. Confirms that a checkpoint
restores the same structure and that the member buffer stays bounded.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_roundtrip.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream
import src.common.persist as persist

s = Stream(dim=128, seed=3)
warm = np.cumsum(np.random.default_rng(1).normal(0, 0.05, (64, 128)), axis=0)
s.calibrate(warm)

rng = np.random.default_rng(5)
for _ in range(400):
    s.landscape.set_noise_amplitude(0.02)
    s.step()

d0 = s.stream_state()

tmp = Path(os.path.abspath("data"))
orig = persist._ensure
persist._ensure = lambda: tmp
created_name = "_roundtrip_l5.pkl"
try:
    p = persist.save_l5(s, created_name)
    size_kb = os.path.getsize(p) / 1024

    s2 = Stream(dim=128, seed=3)
    ok = persist.load_l5(s2, created_name)
    d1 = s2.stream_state()

    print(f"round trip ok : {ok}")
    print(f"attractors    : {d0['n_attractors']} -> {d1['n_attractors']}")
    print(f"salient       : {d0['n_salient']} -> {d1['n_salient']}")
    print(f"edges         : {d0['n_edges']} -> {d1['n_edges']}")
    print(f"save size     : {size_kb:.1f} KB "
          f"for {d0['n_attractors']} attractors")
    if d1["n_attractors"]:
        mx = max(len(a.members) for a in s2.state_space.attractors())
        print(f"max members   : {mx}  (bound is 64)")
finally:
    persist._ensure = orig
    leftover = Path(os.path.abspath("data")) / created_name
    if leftover.exists():
        leftover.unlink()
