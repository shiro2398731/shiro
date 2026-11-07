"""
scripts/_smoke_l348.py

Short smoke test for L3 / L4 / L8 after wiring into Stream.step().
Run it yourself:  .\.venv\Scripts\python.exe scripts\_smoke_l348.py
It uses a synthetic 128-dim driver, no bus, no files.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream

D = 128
N = 900

stream = Stream(dim=D, gamma=0.05, dt=0.01, seed=7)

# zero point: observed median of a warm-up walk
warm = np.cumsum(np.random.default_rng(1).normal(0, 0.05, (64, D)), axis=0)
stream.calibrate(warm)

rng = np.random.default_rng(99)
# a slow random drift so the state actually moves around the space
drift = np.zeros(D)
for i in range(N):
    drift += rng.normal(0, 0.02, D) * 0.3
    stream.landscape.set_noise_amplitude(0.02)
    stream.landscape.set_deflection(drift)
    stream.step()

ss = stream.stream_state()
print("--- after", N, "steps ---")
print("traces        :", ss["n_attractors"])
print("n_salient     :", ss["n_salient"])
print("graph edges   :", ss["n_edges"])
print("tensions      :", ss["tensions"])
print("identity edges:", ss["identity"])
print("recall path   :", ss["recall"])

print("--- L3 ---")
print("graph stats   :", {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in stream.graph.stats().items()})
print("recall describe:", stream._last_recall.describe() if stream._last_recall else "None")
sp = stream.recaller.spontaneous(stream.graph, rng)
print("spontaneous   :", sp.describe())

print("--- L4 ---")
ts = stream.drive.last
print("n tensions    :", len(ts))
for t in ts[:3]:
    print("   ", t.describe())
dv = stream.drive.deflection(stream.state_space, stream.graph)
print("deflection    :", "None" if dv is None else f"norm={np.linalg.norm(dv):.4f}")

print("--- L8 ---")
print("rounds        :", stream.identity.rounds)
print("last core     :", stream._last_core.describe() if stream._last_core else "None")
fp = stream.identity.fingerprint(stream.state_space, stream.graph)
print("fingerprint   :", {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in fp.items()})

print("--- workspace ---")
print("ignitions     :", stream.workspace.ignition_count)
print("baseline      :", stream.workspace.baseline())
