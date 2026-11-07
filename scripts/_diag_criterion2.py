"""
scripts/_diag_criterion2.py

Can a motion-based criterion avoid both extremes?

The extremes, measured:
    per-attractor sigma as criterion -> cold start collapses to ~39 wide
        regions; loaded checkpoint fragments into ~270 regions visited once or
        twice, of which one holds 95% of all assignments.
    accumulated spatial range as criterion -> everything rebuilds (~3000
        regions in 3000 steps).

The accumulator failed because a rolling window over the last N states is
dominated by whether the state happened to be moving. What is tried here is a
criterion based on motion smoothed over a long horizon, compared against a
fraction of the accumulated range.

Nothing in this script writes to the checkpoint; it reimplements assignment
against a fresh space so the criteria can be compared directly.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_criterion2.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.landscape import Landscape
from src.L5_cognition.dynamics import Dynamics, DynamicState
from src.L5_cognition.terrain import StateSpace, Attractor
import src.common.persist as persist
from src.L5_cognition.stream import Stream

D = 128
STEPS = 3000


def trajectory():
    """Reproduce a real trajectory from the checkpoint."""
    s = Stream(dim=D)
    persist.load_l5(s)
    land = s.landscape
    dyn = s.dynamics
    st = s._st.copy()
    rng = np.random.default_rng(7)
    xs = []
    for _ in range(STEPS):
        st = dyn.step(st, land, rng)
        xs.append(st.x.copy())
    return np.stack(xs)


def initial_space():
    sp = StateSpace(dim=D)
    return sp


def run_sigma(X):
    sp = initial_space()
    for x in X:
        sp.record(x)
    return sp


def run_motion(X, frac):
    """
    Criterion: assign to nearest region if the distance is below frac times the
    accumulated range of the trajectory so far. Regions are created with that
    same width.
    """
    sp = initial_space()
    seen = []
    for x in X:
        seen.append(x)
        if len(seen) < 16:
            continue
        M = np.stack(seen)
        c = M.mean(axis=0)
        rng_ = float(np.max(np.linalg.norm(M - c, axis=1)))
        if rng_ <= 0:
            continue
        width = max(rng_ * frac, 1e-12)

        if not sp._attractors:
            a = Attractor(mu=x.copy(), amp=0.05, scale=width)
            a.members.append(x.copy())
            a._assigned = 1
            sp._attractors.append(a)
            sp._invalidate_scale()
            continue

        mus = np.stack([a.mu for a in sp._attractors])
        d = np.linalg.norm(mus - x, axis=1)
        i = int(np.argmin(d))
        if d[i] < width:
            a = sp._attractors[i]
            a.members.append(x.copy())
            a._assigned += 1
            a.amp += (1.0 - a.amp) * 0.05
            a._sigma_cache = None
        else:
            a = Attractor(mu=x.copy(), amp=0.05, scale=width)
            a.members.append(x.copy())
            a._assigned = 1
            sp._attractors.append(a)
            sp._invalidate_scale()
    return sp


def report(label, sp):
    attrs = sp.attractors()
    if not attrs:
        print(f"{label:<24} no regions")
        return
    amps = np.array([a.amp for a in attrs])
    asn = np.array([a._assigned for a in attrs], dtype=float)
    sig = np.array([a.sigma for a in attrs])
    print(f"{label:<24} n={len(attrs):5d}  amp_p50={np.median(amps):.5f}  "
          f"sigma_p50={np.median(sig):.6f}  max_share={asn.max() / asn.sum():.1%}")


X = trajectory()
print(f"trajectory: {len(X)} steps, "
      f"span {np.linalg.norm(X.max(0) - X.min(0)):.2f}")

report("per-attractor sigma", run_sigma(X))
for frac in (0.02, 0.05, 0.1, 0.2):
    report(f"motion range x {frac}", run_motion(X, frac))
