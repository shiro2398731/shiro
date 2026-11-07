"""
Smoke test after the terminology change.

Checks that the pipeline still runs and that the renamed fields appear.
Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_smoke_rename.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream
import src.common.persist as persist

s = Stream(dim=128, seed=7)
warm = np.cumsum(np.random.default_rng(1).normal(0, 0.05, (64, 128)), axis=0)
s.calibrate(warm)

rng = np.random.default_rng(99)
drift = np.zeros(128)
for _ in range(600):
    drift += rng.normal(0, 0.02, 128) * 0.3
    s.landscape.set_noise_amplitude(0.02)
    s.landscape.set_deflection(drift)
    s.step()

d = s.stream_state()
print("n_attractors :", d["n_attractors"])
print("n_salient    :", d["n_salient"])
print("n_edges      :", d["n_edges"])
print("tensions     :", d["tensions"])
print("identity     :", d["identity"])
print("symbols      :", s.get_symbols())

# persist round trip
persist.save_l5(s, "_smoke_rename.pkl")
s2 = Stream(dim=128, seed=7)
ok = persist.load_l5(s2, "_smoke_rename.pkl")
d2 = s2.stream_state()
print("reload ok    :", ok,
      "attractors match:", d["n_attractors"] == d2["n_attractors"])

# cleanup
p = os.path.join("data", "state", "_smoke_rename.pkl")
if os.path.exists(p):
    os.remove(p)
print("done")
