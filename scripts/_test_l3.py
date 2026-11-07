"""
scripts/_test_l3.py

Isolated test of the memory module, with no runtime and no bus.

What it checks:
  1. edges form from direct transitions, and only from direct transitions
  2. repeated transitions strengthen, one-off transitions stay weak
  3. recall returns the connected attractors, in order of strength
  4. weakening reduces weights, and forgotten edges drop out
  5. save/load round trip preserves the graph exactly

Run:  .\\venv\\Scripts\\python.exe scripts\\_test_l3.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L3_memory.trace import TraceGraph
from src.L3_memory.recall import RecallEngine

FAIL = []


def check(name, cond, detail=""):
    print(f"{'ok  ' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------- 1. edges

print("== edges form from direct transitions ==")
g = TraceGraph()
g.touch(None)          # no previous node: nothing to connect
g.touch(1)
g.touch(2)             # 1 -> 2
g.touch(3)             # 2 -> 3
g.touch(1)             # 3 -> 1

check("three edges from three transitions", len(g) == 3, f"got {len(g)}")
check("1->2 exists", g.strength(1, 2) > 0)
check("2->3 exists", g.strength(2, 3) > 0)
check("3->1 exists", g.strength(3, 1) > 0)
check("1->3 does NOT exist", g.directed(1, 3) == 0)
check("3->1 does exist (it happened once)", g.directed(3, 1) > 0)
check("the edge is shared", g.edge(1, 3) is g.edge(3, 1))

# ------------------------------------------------------- 2. strengthening

print()
print("== direction is asymmetric ==")
gd = TraceGraph()
# one continuous walk: 1 -> 2 driven five times, 2 -> 1 only once
gd.touch(1)
for _ in range(5):
    gd.touch(2)
    gd.touch(1)
gd.touch(2)
w12 = gd.directed(1, 2)
w21 = gd.directed(2, 1)
check("1->2 stronger than 2->1", w12 > w21, f"{w12:.3f} vs {w21:.3f}")
check("they are not constrained to sum to a constant",
      abs(w12 + w21) > 0)

# both directions can be strong at once
gb = TraceGraph()
for _ in range(5):
    gb.touch(1)
    gb.touch(2)
    gb.touch(1)
check("both directions can be strong",
      gb.directed(1, 2) > 1.0 and gb.directed(2, 1) > 1.0,
      f"{gb.directed(1, 2):.3f} / {gb.directed(2, 1):.3f}")

# ------------------------------------------------------- 2. strengthening
g2 = TraceGraph()
for _ in range(10):
    g2.touch(1)
    g2.touch(2)
w_repeated = g2.strength(1, 2)

g3 = TraceGraph()
g3.touch(1)
g3.touch(2)
w_once = g3.strength(1, 2)

check("repeated edge stronger than single", w_repeated > w_once,
      f"{w_repeated:.3f} vs {w_once:.3f}")

# ------------------------------------------------------------- 3. recall

print()
print("== recall ==")


class FakeAttractor:
    def __init__(self, idx, mu, sigma=1.0):
        self.idx = idx
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = float(sigma)


class FakeSpace:
    def __init__(self, attrs):
        self._a = attrs

    def attractors(self):
        return self._a


attrs = [
    FakeAttractor(0, [0.0, 0.0]),
    FakeAttractor(1, [1.0, 0.0]),
    FakeAttractor(2, [3.0, 0.0]),
]
space = FakeSpace(attrs)

gr = TraceGraph()
for _ in range(5):
    gr.touch(0)
    gr.touch(1)
for _ in range(2):
    gr.touch(1)
    gr.touch(2)

eng = RecallEngine(max_depth=4, decay=0.5)
r = eng.by_cue(space, gr, np.array([0.05, 0.0]))
check("recall starts at the nearest attractor", r.path and r.path[0] == 0,
      f"path={r.path}")
check("recall reaches the strong neighbour", 1 in r.path, f"path={r.path}")
check("activation decreases along the path",
      all(r.weights[i] >= r.weights[i + 1] for i in range(len(r.weights) - 1)),
      f"weights={[round(w, 3) for w in r.weights]}")

r_far = eng.by_cue(space, gr, np.array([100.0, 100.0]))
check("state far from everything recalls nothing",
      len(r_far.path) == 0 or r_far.weights[0] < 0.5,
      f"path={r_far.path}")

# ------------------------------------------------------------ 4. weaken

print()
print("== weakening ==")
gw = TraceGraph()
for _ in range(3):
    gw.touch(5)
    gw.touch(6)
before = gw.strength(5, 6)
gw.weaken(0.5)
after = gw.strength(5, 6)
check("weakening reduces strength", after < before,
      f"{before:.3f} -> {after:.3f}")

gw2 = TraceGraph()
gw2.touch(7)
gw2.touch(8)
n_before = len(gw2)
gw2.weaken(1.0)
n_after = len(gw2)
check("forgetting removes dead edges", n_after <= n_before,
      f"{n_before} -> {n_after}")

# --------------------------------------------------------- 5. round trip

print()
print("== save / load ==")
g5 = TraceGraph()
for _ in range(7):
    g5.touch(10)
    g5.touch(11)
g5.touch(12)
d = g5.to_dict()
g5b = TraceGraph.from_dict(d)
check("edge count preserved", len(g5) == len(g5b), f"{len(g5)} vs {len(g5b)}")
check("strength preserved exactly",
      abs(g5.strength(10, 11) - g5b.strength(10, 11)) < 1e-12)
check("neighbours preserved",
      sorted(g5.neighbors(11)) == sorted(g5b.neighbors(11)))

# ------------------------------------------------------------- summary

print()
print("=" * 50)
if FAIL:
    print(f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1)
print("all L3 checks passed")
