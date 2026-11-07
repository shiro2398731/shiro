import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import src.common.persist as persist
from src.L5_cognition.stream import Stream
from src.L5_cognition.noise import ErrorNoise
from src.L4_drive.viability import Viability, _interval_scale
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
total = max(int(ss._total_steps), 1)
a = ss.assign_ratio
R0 = ss.recover_rate / max(ss._recover_every_hint, 1)

print("total_steps", total, "assign_ratio", a, "R_base", R0)
print()
print("{:>5}{:>8}{:>10}{:>10}{:>9}{:>9}{:>8}".format(
    "idx", "assign", "life", "p", "amp", "carry", "ratio"))

attrs = ss.attractors()
rows = []
for i, A in enumerate(attrs):
    assigned = float(getattr(A, "_assigned", 0)) or float(len(A.members))
    born = int(getattr(A, "_born_step", 0))
    life = max(total - born, 1)
    p = assigned / life
    car = V.carrying_depth(ss, A, p)
    rows.append((i, int(assigned), life, p, A.amp, car, A.amp / car if car > 0 else 0))
rows.sort(key=lambda t: -t[6])
for r in rows[:6]:
    print("{:>5}{:>8}{:>10}{:>10.6f}{:>9.4f}{:>9.5f}{:>8.1f}".format(*r))
print("...")
for r in rows[-3:]:
    print("{:>5}{:>8}{:>10}{:>10.6f}{:>9.4f}{:>9.5f}{:>8.1f}".format(*r))

print()
print("median life:", int(np.median([r[2] for r in rows])))
print("iss = _interval_scale sample:", round(_interval_scale(ss, attrs[rows[0][0]]), 4))
