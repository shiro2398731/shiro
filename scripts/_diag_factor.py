import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import src.common.persist as persist
from src.L5_cognition.stream import Stream
from src.L5_cognition.noise import ErrorNoise
from src.runtime.process_1_fast import FHNNetwork, L1Predictor


class Up:
    def __init__(self, seed=42):
        self.net = FHNNetwork(seed=seed)
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
for i in range(10000):
    s.landscape.set_noise_amplitude(u.step())
    s.step()

f = s.regulation.last
regs = s.regulation.viability.last
rr = np.array([r.ratio for r in regs])

print("factor distribution over {} regions".format(f.size))
for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
    print("  p{:>3}: {:.4f}".format(q, np.percentile(f, q)))

print()
print("ratio distribution")
for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
    print("  p{:>3}: {:.4f}".format(q, np.percentile(rr, q)))

print()
print("fraction of regions with factor < 0.1: {:.2%}".format(float((f < 0.1).mean())))
print("fraction of regions with factor < 0.01: {:.2%}".format(float((f < 0.01).mean())))

# how much of the total pull does the weakest half contribute
order = np.argsort(f)
half = f.size // 2
print()
print("sum of factor, weakest half : {:.2f}".format(float(f[order[:half]].sum())))
print("sum of factor, strongest half: {:.2f}".format(float(f[order[half:]].sum())))
