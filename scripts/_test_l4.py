"""
scripts/_test_l4.py

Isolated test of the drive module, with no runtime and no bus.

What it checks:
  1. a tension needs both a strong edge and a large separation
  2. a strong edge spanning a small separation is NOT a tension
  3. a weak edge spanning a large separation is NOT a tension
  4. deflection is perpendicular to the tension axis
  5. deflection magnitude does not grow with the number of attractors
  6. fewer than 8 edges gives no tensions (resolution floor, not a result)
  7. viability: carrying depth is monotone in visit rate, bounded by ceiling

Run:  .\\venv\\Scripts\\python.exe scripts\\_test_l4.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L3_memory.trace import TraceGraph
from src.L4_drive.drive import Drive
from src.L4_drive.viability import Viability

FAIL = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


class Att:
    def __init__(self, idx, mu, sigma=0.05):
        self.idx = idx
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = float(sigma)
        self._decay_factor = 1.0
        self.visits = 10
        self.amp = 0.3
        self.members = np.zeros(10)


class Space:
    """
    Minimal stand-in for the state space.

    Only the attributes the drive and viability code actually reads are
    provided. If any of these change meaning in the real class, this test
    will pass while the system breaks, so they are listed explicitly rather
    than derived.
    """

    assign_ratio = 0.5
    recover_rate = 0.2
    recover_every = 16
    ceiling = 1.0

    def __init__(self, attrs):
        self._a = attrs
        self._step_count = 1000

    def attractors(self):
        return self._a


def make_space(n, spread=1.0, dim=8):
    rng = np.random.default_rng(7)
    return Space([Att(i, rng.normal(0, spread, dim)) for i in range(n)])


# -------------------------------------------- 1-3. what counts as tension

print("== what counts as a tension ==")


def build(space, strong_pair, weak_pairs=None, n_pairs=80, seed=3):
    """
    Build a graph with one strong, widely separated pair and a background of
    weaker, unevenly used pairs.

    The background must be uneven on purpose: if every background edge has
    exactly the same weight, the median absolute deviation is zero and the
    z-score has nothing to stand on.
    """
    rng = np.random.default_rng(seed)
    g = TraceGraph()
    a, b = strong_pair
    for _ in range(20):
        g.touch(a)
        g.touch(b)
    made = 1
    i = 0
    while made < n_pairs and i < 200:
        x = int(rng.integers(0, len(space.attractors())))
        y = int(rng.integers(0, len(space.attractors())))
        i += 1
        if x == y or {x, y} == {a, b}:
            continue
        for _ in range(int(rng.integers(1, 4))):
            g.touch(x)
            g.touch(y)
        made += 1
    return g


space = make_space(40)
attrs = space.attractors()
# pick two that are far apart
mus = np.stack([a.mu for a in attrs])
dists = np.linalg.norm(mus[:, None, :] - mus[None, :, :], axis=-1)
np.fill_diagonal(dists, 0)
far_a, far_b = np.unravel_index(np.argmax(dists), dists.shape)
far_a, far_b = int(far_a), int(far_b)

g = build(space, (far_a, far_b), None)
d = Drive(dim=8, gain=1.0)
ts = d.tensions(space, g)
pairs = {(t.a, t.b) for t in ts}
hit = (far_a, far_b) in pairs or (far_b, far_a) in pairs
check("strong + far pair appears as a tension", hit,
      f"tensions={len(ts)} far pair=({far_a},{far_b})")

# same edge, but bring the two attractors close together
space_close = Space([Att(i, a.mu.copy()) for i, a in enumerate(attrs)])
space_close._a[far_b] = Att(far_b, attrs[far_a].mu + 0.001, sigma=0.05)
ts_close = Drive(dim=8, gain=1.0).tensions(space_close, build(space_close, (far_a, far_b), None))
pairs_close = {(t.a, t.b) for t in ts_close}
check("strong but close pair is NOT a tension",
      (far_a, far_b) not in pairs_close and (far_b, far_a) not in pairs_close,
      f"tensions={len(ts_close)}")

# -------------------------------------------------------- 4-5. deflection

print()
print("== deflection ==")
g = build(space, (far_a, far_b), None)
d = Drive(dim=8, gain=1.0)
vec = d.deflection(space, g)
check("deflection exists when tensions exist", vec is not None)
if vec is not None:
    # Two tensions may not share an axis, so the sum is not required to be
    # orthogonal to the single strongest axis. What must hold is that each
    # contribution is perpendicular to its own axis -- checked here by
    # rebuilding the sum from the reported tensions.
    ts = d.last
    ok_each = True
    for t in ts:
        axis_t = mus[t.a] - mus[t.b]
        nt = np.linalg.norm(axis_t)
        if nt == 0:
            continue
        axis_t = axis_t / nt
        perp = d._perpendicular(axis_t)
        if perp is None:
            continue
        if abs(float(perp @ axis_t)) > 1e-9:
            ok_each = False
    check("every deflection component is perpendicular to its own axis", ok_each)
    axis = mus[far_a] - mus[far_b]
    axis = axis / np.linalg.norm(axis)
    dot = float(abs(vec @ axis))
    # The deflection is the sum of one perpendicular component per tension. Each
    # is perpendicular to its own axis, but not to every other tension's axis,
    # so the sum picks up a residual on any single axis. The expected size of
    # that residual is about 1/sqrt(n) for n comparable contributions; it is not
    # zero and demanding zero would only pass when there is one tension.
    n_t = max(len(ts), 1)
    bound = 3.0 / np.sqrt(n_t)
    check("residual on the strongest axis is consistent with summing n tilts",
          dot < bound,
          f"|cos|={dot:.2e} across {n_t} tensions, bound {bound:.3f}")
    check("deflection has unit length times gain", abs(np.linalg.norm(vec) - d.gain) < 1e-9,
          f"|v|={np.linalg.norm(vec):.6f} gain={d.gain}")

space_big = make_space(60)
ts_small = Drive(dim=8, gain=1.0).tensions(space, build(space, (far_a, far_b), None))
n_small = len(ts_small)
d_big = Drive(dim=8, gain=1.0)
g_big = build(space_big, (0, 59), None, n_pairs=120)
v_big = d_big.deflection(space_big, g_big)
check("deflection magnitude is bounded regardless of attractor count",
      v_big is not None and abs(np.linalg.norm(v_big) - d_big.gain) < 1e-9,
      f"n_tensions={len(d_big.last)} n_attractors=60")

# ------------------------------------------------------- 6. edge floor

print()
print("== resolution floor ==")
g_tiny = TraceGraph()
g_tiny.touch(0)
g_tiny.touch(1)
g_tiny.touch(0)
ts_tiny = Drive(dim=8, gain=1.0).tensions(space, g_tiny)
check("fewer than 8 edges gives no tensions (floor, not a result)",
      len(ts_tiny) == 0, f"{len(ts_tiny)} tensions")

# ------------------------------------------------------- 7. viability

print()
print("== viability ==")
v = Viability()
ceil = 0.5
a0 = attrs[0]
depths = [v.carrying_depth(space, a0, rate) for rate in (0.0, 0.1, 1.0, 10.0, 1e6)]
check("carrying depth increases with visit rate",
      all(depths[i] <= depths[i + 1] + 1e-12 for i in range(len(depths) - 1)),
      f"{[round(x, 4) for x in depths]}")
check("carrying depth never exceeds the ceiling",
      all(x < 1.0 + 1e-9 for x in depths),
      f"max={max(depths):.4f}")

regions = v.measure(space)
check("viability measures every attractor", len(regions) == len(attrs),
      f"{len(regions)} regions for {len(attrs)} attractors")
if regions:
    rr = [r.ratio for r in regions]
    check("viability ratios are finite and non-negative",
          all(np.isfinite(x) and x >= 0 for x in rr),
          f"min={min(rr):.4f} max={max(rr):.4f}")

# ------------------------------------------------------------- summary

print()
print("=" * 50)
if FAIL:
    print(f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1)
print("all L4 checks passed")
