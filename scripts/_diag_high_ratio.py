import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import src.common.persist as persist
from src.L5_cognition.stream import Stream
from src.L5_cognition.noise import ErrorNoise
from src.L4_drive.viability import Viability
from src.runtime.process_1_fast import FHNNetwork, L1Predictor


class Up:
    def __init__(self):
        self.net = FHNNetwork(seed=42)
        self.pred = L1Predictor(input_dim=100)
        self.n = ErrorNoise(dim=128)

    def step(self):
        sn = self.net.step()
        self.pred.step(sn["E_rate_vec"])
        return self.n.observe(self.pred.last_error, self.pred.proj)


s = Stream(dim=128)
persist.load_l5(s)
try:
    s.step()
except RuntimeError:
    s.calibrate(np.random.default_rng(1).normal(0.5, 0.1, (64, 128)))

u = Up()
for i in range(6000):
    s.landscape.set_noise_amplitude(u.step())
    s.step()

ss = s.landscape.state_space
V = Viability()
attrs = ss.attractors()
rows = []
for i, A in enumerate(attrs):
    p = A._assigned / max(ss._total_steps, 1)
    car = V.carrying_depth(ss, A, p)
    r = A.amp / car if car > 0 else float("inf")
    members = len(A.members)
    rows.append((i, A._assigned, members, A.amp, car, r))
rows.sort(key=lambda t: -t[5])

print("{:>5}{:>8}{:>9}{:>9}{:>9}{:>8}".format(
    "idx", "assign", "members", "amp", "carry", "ratio"))
print("-- highest ratio --")
for r in rows[:8]:
    print("{:>5}{:>8}{:>9}{:>9.5f}{:>9.5f}{:>8.2f}".format(*r))
print("-- lowest ratio --")
for r in rows[-5:]:
    print("{:>5}{:>8}{:>9}{:>9.5f}{:>9.5f}{:>8.2f}".format(*r))

hi = [r for r in rows if r[5] > 2.0]
print()
print("regions with ratio > 2: {}".format(len(hi)))
if hi:
    print("  their assigned counts: min={} median={} max={}".format(
        min(r[1] for r in hi),
        int(np.median([r[1] for r in hi])),
        max(r[1] for r in hi)))
    print("  their amps: min={:.5f} median={:.5f} max={:.5f}".format(
        min(r[3] for r in hi),
        float(np.median([r[3] for r in hi])),
        max(r[3] for r in hi)))
    print("  their carrying: min={:.5f} median={:.5f} max={:.5f}".format(
        min(r[4] for r in hi),
        float(np.median([r[4] for r in hi])),
        max(r[4] for r in hi)))
