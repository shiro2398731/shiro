"""
scripts/_diag_escape.py

How strong is the attraction compared with the fluctuation that has to
overcome it?

For a Gaussian attractor the gradient at distance r from the centroid is

    |grad| = amp * exp(-r^2 / (2 sigma^2)) * r / sigma^2

It peaks at r = sigma, where |grad| = amp / (sigma * sqrt(e)).

The velocity increment from fluctuation in one step is noise * dt.
Comparing the two says whether escape is possible at all, and if so over
what time scale.

This uses ONLY the real checkpoint's parameters. Nothing is simulated.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_escape.py
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

PATH = os.path.join("data", "state", "l5.pkl")
with open(PATH, "rb") as f:
    obj = pickle.load(f)

raw = obj.get("attractors") or obj.get("traces") or []

amps = np.array([r["amp"] for r in raw], dtype=float)
sig = []
for r in raw:
    m = np.stack([np.asarray(x, dtype=float) for x in r["members"]]) if r["members"] else None
    if m is not None and len(m) >= 4:
        d = m - m.mean(axis=0)
        sig.append(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))))
    else:
        sig.append(float(r["scale"] or 1.0))
sig = np.array(sig)

order = np.argsort(-amps)
i = order[0]

print("=== deepest attractor ===")
print(f"  amp   : {amps[i]:.6f}")
print(f"  sigma : {sig[i]:.6f}")
print(f"  members: {len(raw[i]['members'])}")

peak_grad = amps[i] / (sig[i] * np.sqrt(np.e))
print(f"  peak |grad| at r = sigma : {peak_grad:.3f}")

# gradient right at the edge of the assignment zone (r = sigma)
g_at_sigma = amps[i] * np.exp(-0.5) * sig[i] / (sig[i] ** 2)
print(f"  |grad| at r = sigma      : {g_at_sigma:.3f}")

# parameters used by the integrator
DT = 0.01
GAMMA = 0.05
print(f"\n=== integrator ===")
print(f"  dt    : {DT}")
print(f"  gamma : {GAMMA}")

print("\n=== noise amplitudes actually observed in the log ===")
LOG = os.path.join("data", "logs", "observations.jsonl")
noises = []
if os.path.exists(LOG):
    import json
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            if '"section": "L5"' not in line:
                continue
            try:
                d = json.loads(line)["data"]
            except Exception:
                continue
            if "noise" in d and d["noise"] is not None:
                noises.append(float(d["noise"]))
noises = np.array(noises)
if noises.size:
    print(f"  median {np.median(noises):.5f}   p90 {np.percentile(noises, 90):.5f}   "
          f"max {noises.max():.5f}")
else:
    print("  no log data")

noise = float(np.median(noises)) if noises.size else 0.01

print("\n=== can the fluctuation overcome the attraction? ===")
v_per_step = noise * DT
print(f"  velocity added per step by noise : {v_per_step:.8f}")
print(f"  velocity needed to escape peak gradient region : {peak_grad * DT:.8f}")
ratio = v_per_step / (peak_grad * DT)
print(f"  ratio (noise / gradient) : {ratio:.3e}")

# terminal velocity under constant gradient: v = g / gamma
v_terminal = peak_grad / GAMMA
print(f"\n  terminal drift speed under this gradient : {v_terminal:.3f}")
print(f"  distance to travel to leave the region   : {sig[i]:.6f}")
print(f"  time at terminal speed                   : {sig[i] / v_terminal:.3e} steps")

# how many sigma away would noise random walk take to travel one sigma
# random walk: sigma_rw(n) = v_per_step * sqrt(n)  => n = (sigma / v_per_step)^2
n = (sig[i] / v_per_step) ** 2
print(f"\n  random-walk steps to cover one sigma : {n:.3e}")
