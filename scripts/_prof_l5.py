import sys, os, cProfile, pstats, io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import src.common.persist as persist
from src.L5_cognition.stream import Stream
from src.L5_cognition.noise import ErrorNoise
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
# warm up so the profile is not dominated by one-off setup
for _ in range(2000):
    s.landscape.set_noise_amplitude(u.step())
    s.step()


def workload():
    for _ in range(2000):
        s.landscape.set_noise_amplitude(u.step())
        s.step()


pr = cProfile.Profile()
pr.enable()
workload()
pr.disable()

st = pstats.Stats(pr, stream=io.StringIO())
st.sort_stats("cumulative")
buf = io.StringIO()
st.stream = buf
st.print_stats(28)
print(buf.getvalue())
