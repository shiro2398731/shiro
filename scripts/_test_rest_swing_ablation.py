import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import src.common.persist as persist
from src.L5_cognition.stream import Stream
from src.L5_cognition.noise import ErrorNoise
from src.L5_cognition.rest import RestField
from src.L5_cognition.swing import Swing
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


class DeadRest(RestField):
    def step(self, occupied, rest_rate_scale=1.0):
        return


class DeadSwing(Swing):
    def step(self, x):
        return

    def tilt(self):
        return None


def run(use_rest, use_swing, steps=8000):
    s = Stream(dim=128)
    persist.load_l5(s)
    try:
        s.step()
    except RuntimeError:
        s.calibrate(np.random.default_rng(1).normal(0.5, 0.1, (64, 128)))
    if not use_rest:
        s.rest = DeadRest()
    if not use_swing:
        s.swing = DeadSwing(dim=128)

    u = Up()
    speeds = []
    trans = 0
    for _ in range(steps):
        s.landscape.set_noise_amplitude(u.step())
        ev = s.step()
        if ev is not None:
            trans += 1
        speeds.append(s.stream_state()["speed"])

    ss = s.stream_state()
    g = s.graph.stats()
    return (ss["n_attractors"], g["edges"], float(np.mean(speeds)),
            float(np.std(speeds)), trans, g["asymmetry"])


print("{:>8}{:>8}{:>6}{:>7}{:>10}{:>10}{:>7}{:>8}".format(
    "rest", "swing", "att", "edges", "speed", "speed_sd", "trans", "asym"))
for use_rest in (True, False):
    for use_swing in (True, False):
        a, e, sp, sd, tr, asy = run(use_rest, use_swing)
        print("{:>8}{:>8}{:>6}{:>7}{:>10.4f}{:>10.4f}{:>7}{:>8.3f}".format(
            "on" if use_rest else "off", "on" if use_swing else "off",
            a, e, sp, sd, tr, asy))
