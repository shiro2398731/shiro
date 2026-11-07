"""
scripts/_test_regulation.py

Test the regulation layer.

Regulation turns the viability ratio of each region into a pull factor. The
ratio is the region's depth divided by the depth it can sustain at its observed
visit rate; a factor of 1 means the region is holding exactly as much as it can,
below 1 means it is underfilled, above 1 means it is carrying more than its
visit rate supports.

There is no ceiling on the factor. One used to exist at 4.0, and it was there to
hide a miscomputed ratio rather than to express a requirement: with the visit
rate measured over the region's own lifetime instead of the whole run, the
population maximum is about 1.35, and sweeping the cap over 2, 4, 8 and none
changed nothing because no factor ever reached it.

What this checks:
  1. the factor is finite and non-negative
  2. at beta = 1 the factor equals the viability ratio exactly (identity)
  3. beta orders the factors: spread grows with beta
  4. a region further below its carrying level has a smaller factor
  5. factor 1.0 means the region is exactly at what it can carry

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_regulation.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L4_drive.regulation import Regulation

FAIL = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


class Att:
    def __init__(self, idx, mu, sigma=0.05, visits=10, born=0, amp=0.3):
        self.idx = idx
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = float(sigma)
        self._decay_factor = 1.0
        self.visits = visits
        self.amp = float(amp)
        self.members = np.zeros(max(visits, 1))
        self._assigned = visits
        # Visit rate is measured over the region's own lifetime, so the born
        # step has to be a real value here: leaving it at zero makes every
        # region look as old as the run.
        self._born_step = int(born)


class Space:
    assign_ratio = 0.5
    recover_rate = 0.2
    recover_every = 16
    ceiling = 1.0

    def __init__(self, attrs, total_steps=1000):
        self._a = attrs
        self._total_steps = total_steps
        self._step_count = total_steps
        self._recover_every_hint = 16

    def attractors(self):
        return self._a


rng = np.random.default_rng(11)
attrs = []
for i in range(20):
    v = int(rng.integers(1, 40))
    attrs.append(Att(i, rng.normal(0, 1, 8), visits=v, born=0, amp=0.3))
space = Space(attrs, total_steps=1000)

# ------------------------------------------------------- 1. bounds

print("== bounds ==")
for beta in (0.0, 0.5, 1.0, 2.0):
    reg = Regulation(beta=beta)
    f = reg.factors(space)
    check(f"beta={beta}: factors are finite and >= 0",
          bool(np.all(np.isfinite(f)) and np.all(f >= 0)),
          f"min={f.min():.4f} max={f.max():.4f}")

# -------------------------------------------------- 2. identity at beta=1

print()
print("== beta = 1 is the identity relation ==")
reg1 = Regulation(beta=1.0)
f1 = reg1.factors(space)
# measure() returns regions sorted by ratio, not in attractor order, so the
# comparison has to go through r.index.
ratios = np.zeros_like(f1)
for r in reg1.viability.last:
    ratios[r.index] = r.ratio
if ratios.size == f1.size:
    err = float(np.max(np.abs(f1 - ratios)))
    check("factor equals viability ratio exactly, index-aligned", err < 1e-9,
          f"max |f - ratio| = {err:.2e}")
else:
    check("factor equals viability ratio exactly, index-aligned", False,
          f"size mismatch {f1.size} vs {ratios.size}")

# --------------------------------------------- 3. monotone in beta

print()
print("== beta orders the factors ==")
spreads = {}
for beta in (0.0, 0.5, 1.0, 2.0):
    reg = Regulation(beta=beta)
    f = reg.factors(space)
    spreads[beta] = float(np.std(f))
    print(f"   beta={beta:<4} mean={f.mean():.4f} std={f.std():.4f} "
          f"max={f.max():.4f}")

check("beta=0 makes every factor equal (no regulation)",
      spreads[0.0] < 1e-9, f"std={spreads[0.0]:.2e}")
check("spread increases with beta",
      spreads[0.5] < spreads[1.0] < spreads[2.0],
      f"{spreads[0.5]:.4f} < {spreads[1.0]:.4f} < {spreads[2.0]:.4f}")

# ------------------------------------------------------- 4. direction

print()
print("== the pull direction ==")
reg_a = Regulation(beta=1.0)
fa = reg_a.factors(space)
rr = np.zeros_like(fa)
for r in reg_a.viability.last:
    rr[r.index] = r.ratio
if rr.size == fa.size and rr.size > 2:
    hi = np.argsort(rr)[-3:]
    lo = np.argsort(rr)[:3]
    check("lower viability ratio gives a lower factor",
          fa[lo].mean() < fa[hi].mean(),
          f"low ratio factor={fa[lo].mean():.4f} high ratio factor={fa[hi].mean():.4f}")

# ------------------------------------------ 5. what a factor of one means

print()
print("== factor 1.0 means the region carries what its visits support ==")
# A region whose depth exactly equals its carrying depth should come out at
# 1.0. Construct it by asking viability what it can carry at a known rate and
# then setting the depth to that value.
probe_space = Space([Att(0, np.zeros(8), visits=100, born=0, amp=0.0)], total_steps=1000)
reg_p = Regulation(beta=1.0)
car = reg_p.viability.carrying_depth(probe_space, probe_space.attractors()[0],
                                     100 / 1000.0)
probe_space.attractors()[0].amp = car
fp = reg_p.factors(probe_space)
check("a region at exactly its carrying depth has factor 1.0",
      abs(float(fp[0]) - 1.0) < 1e-9, f"factor={float(fp[0]):.6f} carrying={car:.6f}")

# ---------------------------------------------------------- summary

print()
print("=" * 55)
if FAIL:
    print(f"{len(FAIL)} FAILED:")
    for x in FAIL:
        print(f"  - {x}")
    sys.exit(1)
print("all regulation checks passed")
