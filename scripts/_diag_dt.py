"""
scripts/_diag_dt.py

Does dt alone explain why the state never leaves its attractor?

Method: load the REAL saved terrain, then free-run from her real position
with several dt values. Report how far she actually travels, measured
in units of her own territory radius (not in absolute numbers, because
absolute numbers mean nothing without her own scale).

Everything is measured relative to her own terrain. No synthetic driver.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_dt.py
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.landscape import Landscape
from src.L5_cognition.dynamics import Dynamics, DynamicState
from src.L5_cognition.terrain import Terrain, Trace

PATH = os.path.join("data", "state", "l5.pkl")

with open(PATH, "rb") as f:
    obj = pickle.load(f)

traces = obj["traces"]
zero = np.asarray(obj["zero"], dtype=float)
x0 = np.asarray(obj["x"], dtype=float)
step_count = obj["step_count"]
gscale = obj["global_scale"]

print("--- her real state ---")
print(f"step_count   : {step_count}")
print(f"global_scale : {gscale:.6f}")

# ---- rebuild a real Terrain from the saved traces ----
terr = Terrain(dim=128)
for r in traces:
    t = Trace(
        mu=np.asarray(r["mu"], dtype=float),
        amp=float(r["amp"]),
        born_t=r.get("born_t"),
        scale=float(r["scale"]) if r.get("scale") is not None else None,
    )
    t.members.extend([np.asarray(m, dtype=float) for m in r["members"]])
    t._sigma_decay = r.get("_sigma_decay")
    t._sigma_cache = None
    terr._traces.append(t)

sig = np.array([t.sigma for t in terr.traces()], dtype=float)
amps = np.array([t.amp for t in terr.traces()], dtype=float)
print(f"traces       : {len(traces)}")
print(f"sigma p10/50/90 : {np.percentile(sig,10):.6f} / "
      f"{np.percentile(sig,50):.6f} / {np.percentile(sig,90):.6f}")
print(f"amp   p50/90/max: {np.percentile(amps,50):.6f} / "
      f"{np.percentile(amps,90):.6f} / {amps.max():.6f}")

# her typical territory radius, in the space she actually lives in
R = float(np.median(sig))

# ---- noise amplitude: what the real run had ----
NOISE = 0.015

print("\n--- free run: how many territory radii does she travel? ---")
print("(simulated for a fixed number of integration steps; the projection")
print(" is what she would cover starting from where she is right now)")
print()
print(f"{'dt':>8} {'gamma':>7} {'steps':>7} {'span/R':>8} {'final/R':>8} "
      f"{'moved?':>7}")

for gamma in (0.05,):
    for dt in (0.01, 0.03, 0.1, 0.3, 1.0):
        land = Landscape(dim=128)
        land._zero = zero.copy()
        land.terrain = terr
        land.set_noise_amplitude(NOISE)

        dyn = Dynamics(dim=128, gamma=gamma, dt=dt)
        st = DynamicState(x=x0.copy(), v=np.zeros(128))
        rng = np.random.default_rng(3)

        n_steps = 4000
        far = 0.0
        for _ in range(n_steps):
            st = dyn.step(st, land, rng)
            d = float(np.linalg.norm(st.x - x0))
            if d > far:
                far = d

        final = float(np.linalg.norm(st.x - x0))
        moved = "yes" if final / R > 1.0 else "no"
        print(f"{dt:8.3f} {gamma:7.3f} {n_steps:7d} {far/R:8.3f} "
              f"{final/R:8.3f} {moved:>7}")

print()
print("span/R = furthest distance reached, in units of her territory radius")
print("final/R = distance from start at the end, same units")
print("'moved' = yes means she ended up at least one territory away")
