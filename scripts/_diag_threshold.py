import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from src.L3_memory.trace import TraceGraph
from src.L4_drive.drive import Drive


class Att:
    def __init__(self, idx, mu, sigma=0.05):
        self.idx = idx
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = float(sigma)


class Space:
    def __init__(self, attrs):
        self._a = attrs

    def attractors(self):
        return self._a


# How many tensions does the same structure produce as more edges are added?
# If the answer keeps changing with the edge count, the threshold is deciding
# the result rather than measuring it.
rng = np.random.default_rng(5)
attrs = [Att(i, rng.normal(0, 1, 8)) for i in range(40)]
space = Space(attrs)

print("{:>7}{:>10}{:>10}{:>12}".format("edges", "tensions", "mean_amt", "max_amt"))
prev = None
for target in (4, 8, 16, 32, 64, 128, 256, 400):
    g = TraceGraph()
    made = 0
    i = 0
    while made < target and i < target * 20:
        a = int(rng.integers(0, 40))
        b = int(rng.integers(0, 40))
        i += 1
        if a == b:
            continue
        # Vary how often each pair is walked, so the weight distribution has a
        # spread rather than being uniform.
        for _ in range(int(rng.integers(1, 8))):
            g.touch(a)
            g.touch(b)
        made += 1
    d = Drive(dim=8, gain=1.0)
    ts = d.tensions(space, g)
    amts = [t.amount for t in ts] if ts else [0.0]
    print("{:>7}{:>10}{:>10.3f}{:>12.3f}".format(
        len(g), len(ts), float(np.mean(amts)), float(np.max(amts))))
