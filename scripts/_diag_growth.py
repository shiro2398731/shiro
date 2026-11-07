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
V = Viability()
ss = s.landscape.state_space

# Watch a handful of regions over time instead of the whole population.
watch = [0, 5, 20, 100]

print("{:>6}{:>8}{:>9}{:>9}{:>9}{:>9}".format(
    "step", "region", "assign", "amp", "carry", "ratio"))
for i in range(6000):
    s.landscape.set_noise_amplitude(u.step())
    s.step()
    if (i + 1) % 1500 == 0:
        attrs = ss.attractors()
        for w in watch:
            if w < len(attrs):
                A = attrs[w]
                p = A._assigned / max(ss._total_steps, 1)
                car = V.carrying_depth(ss, A, p)
                r = A.amp / car if car > 0 else 0
                print("{:>6}{:>8}{:>9}{:>9.4f}{:>9.4f}{:>9.2f}".format(
                    i + 1, w, A._assigned, A.amp, car, r))

# Whole-population summary at the end
regs = V.measure(ss)
r = np.array([x.ratio for x in regs])
print()
print("population: n={} median={:.2f} max={:.2f}".format(r.size, np.median(r), r.max()))
amps = np.array([A.amp for A in ss.attractors()])
print("amp: median={:.4f} max={:.4f}".format(np.median(amps), amps.max()))
print("total_steps", ss._total_steps)
