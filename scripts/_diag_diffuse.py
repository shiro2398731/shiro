"""
scripts/_diag_diffuse.py

At the bottom of an attractor the gradient is zero, so the state is simply
diffusing. How far should diffusion carry it in the number of steps actually
observed, and how far did it actually go?

This compares the model's prediction with the real checkpoint, using the
real parameters. No simulation of behaviour beyond the integrator itself.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_diffuse.py
"""

import json
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

i = int(np.argmax(amps))
members = np.stack([np.asarray(x, dtype=float) for x in raw[i]["members"]])
mu = members.mean(axis=0)
print("=== deepest attractor, from its own members ===")
print(f"  amp     : {amps[i]:.6f}")
print(f"  sigma   : {sig[i]:.6f}")
print(f"  members : {len(members)}")
r = np.linalg.norm(members - mu, axis=1)
print(f"  member radius: median {np.median(r):.6f}  max {r.max():.6f}")
print(f"  implied: the state genuinely stays within ~{np.median(r):.4f} of the centroid")

print("\n=== diffusion prediction ===")
DT = 0.01
GAMMA = 0.05
NOISE = 0.01001   # median observed

# Under dv/dt = -gamma v + eta, the velocity is an OU process with
# stationary variance = noise^2 / (2 gamma) per component.
# Displacement over n steps: variance = v_var * dt^2 * n (rough diffusion)
v_var = NOISE ** 2 / (2 * GAMMA)
print(f"  stationary velocity variance per component : {v_var:.6e}")
print(f"  rms speed per component                    : {np.sqrt(v_var):.6f}")
print(f"  speed norm (128 dims)                      : {np.sqrt(v_var * 128):.6f}")

for n in (1000, 10000, 100000):
    # displacement grows as sqrt(n) * dt * rms_component_speed * sqrt(dims)
    disp = np.sqrt(n) * DT * np.sqrt(v_var) * np.sqrt(128)
    print(f"  predicted displacement after {n:>6} steps : {disp:.6f}")

print(f"\n  observed member radius (median) : {np.median(r):.6f}")
print(f"  attractor sigma                 : {sig[i]:.6f}")

print("\n=== conclusion check ===")
print("  If predicted displacement >> sigma, the state should have left.")
print("  If comparable, it is marginal.")
print("  If smaller, attraction keeps pulling it back before it can escape.")
