"""
scripts/_diag_criterion.py

What happens if the assignment criterion stops using the attractor's own
characteristic scale?

Compares several candidate criteria on the same real trajectory window.
Nothing is written to the checkpoint.

Criteria:
  A. current : assign if ||x - mu|| / sigma_a < 1   (sigma per attractor)
  B. global  : assign if ||x - mu|| / global_scale < k, for several k

The number that matters is not "how many attractors exist" but whether
concentration stays bounded instead of collapsing into one dominant region.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_criterion.py
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.landscape import Landscape
from src.L5_cognition.dynamics import Dynamics, DynamicState
from src.L5_cognition.terrain import StateSpace, Attractor

PATH = os.path.join("data", "state", "l5.pkl")

with open(PATH, "rb") as f:
    obj = pickle.load(f)

raw = obj.get("attractors") or obj.get("traces") or []
zero = np.asarray(obj["zero"], dtype=float)
x0 = np.asarray(obj["x"], dtype=float)
gscale = float(obj["global_scale"])

print(f"saved state: {len(raw)} attractors, global_scale={gscale:.6f}")


def rebuild(space):
    for r in raw:
        a = Attractor(
            mu=np.asarray(r["mu"], dtype=float),
            amp=float(r["amp"]),
            scale=float(r["scale"]) if r.get("scale") is not None else None,
        )
        a.members.extend([np.asarray(m, dtype=float) for m in r["members"]])
        a._decay_factor = r.get("_decay_factor", r.get("_sigma_decay"))
        a._sigma_cache = None
        space._attractors.append(a)
    space._global_scale = gscale
    space._recent.extend([np.asarray(m, dtype=float) for m in obj["recent"]])


# ---------- reproduce a trajectory window ----------

land = Landscape(dim=128)
land._zero = zero.copy()
rebuild(land.state_space)
land.set_noise_amplitude(0.015)

dyn = Dynamics(dim=128, gamma=0.05, dt=0.01)
st = DynamicState(x=x0.copy(), v=np.asarray(obj["v"], dtype=float))
rng = np.random.default_rng(11)

N = 3000
xs = []
for _ in range(N):
    st = dyn.step(st, land, rng)
    xs.append(st.x.copy())
X = np.stack(xs)

print(f"\ntrajectory window: {N} steps")
disp = np.linalg.norm(np.diff(X, axis=0), axis=1)
print(f"step displacement: median {np.median(disp):.6f}  "
      f"p90 {np.percentile(disp, 90):.6f}")
print(f"span of window   : {np.linalg.norm(X.max(0) - X.min(0)):.6f}")
print(f"global_scale     : {gscale:.6f}")


def report(label, space):
    counts = np.array([len(a.members) for a in space._attractors], dtype=float)
    if counts.size == 0:
        print(f"  {label:<22} attractors=0")
        return
    total = counts.sum()
    share = counts.max() / total
    print(f"  {label:<22} attractors={len(counts):>5}  "
          f"largest share={share:>6.1%}  "
          f"regions with >1%={int((counts / total > 0.01).sum()):>4}")


def run_current():
    space = StateSpace(dim=128)
    space._global_scale = gscale
    space._recent.extend([np.asarray(m, dtype=float) for m in obj["recent"]])
    for x in X:
        space.record(x)
    report("current (sigma_a)", space)
    sig = np.array([a.sigma for a in space._attractors], dtype=float)
    print(f"  {'':22} sigma range {sig.min():.6f} .. {sig.max():.6f}")


def make_assign_global(k):
    def f(xs):
        space = StateSpace(dim=128)
        space._global_scale = gscale
        for x in xs:
            if not space._attractors:
                a = Attractor(mu=x.copy(), amp=0.05, scale=gscale)
                a.members.append(x.copy())
                space._attractors.append(a)
                continue
            mus = np.stack([a.mu for a in space._attractors])
            d = np.linalg.norm(mus - x, axis=1) / gscale
            i = int(np.argmin(d))
            if d[i] < k:
                a = space._attractors[i]
                a.members.append(x.copy())
                n = len(a.members)
                a.mu += (x - a.mu) / n
                a._sigma_cache = None
            else:
                a = Attractor(mu=x.copy(), amp=0.05, scale=gscale)
                a.members.append(x.copy())
                space._attractors.append(a)
        report(f"global x {k}", space)
        sig = np.array([a.sigma for a in space._attractors], dtype=float)
        if sig.size:
            print(f"  {'':22} sigma range {sig.min():.6f} .. {sig.max():.6f}")
    return f


print("\n--- A: per-attractor sigma (current behaviour) ---")
run_current()

print("\n--- B: global scale criterion ---")
for k in (1, 2, 5, 10, 20, 50):
    make_assign_global(k)(X)
