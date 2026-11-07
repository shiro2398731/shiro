import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import src.common.persist as persist
from src.L5_cognition.stream import Stream
from src.L5_cognition.noise import ErrorNoise
from src.L4_drive.regulation import Regulation
from src.runtime.process_1_fast import FHNNetwork, L1Predictor


class Up:
    """Deterministic upstream, so two runs see identical input."""

    def __init__(self, seed=42):
        self.net = FHNNetwork(seed=seed)
        self.pred = L1Predictor(input_dim=100)
        self.n = ErrorNoise(dim=128)

    def step(self):
        sn = self.net.step()
        self.pred.step(sn["E_rate_vec"])
        return self.n.observe(self.pred.last_error, self.pred.proj)


def run(ceiling, steps=10000):
    s = Stream(dim=128)
    persist.load_l5(s)
    try:
        s.step()
    except RuntimeError:
        s.calibrate(np.random.default_rng(1).normal(0.5, 0.1, (64, 128)))
    s.regulation = Regulation(ceiling=ceiling)

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
    f = s.regulation.last
    regs = s.regulation.viability.last
    rr = np.array([r.ratio for r in regs])
    return {
        "att": ss["n_attractors"],
        "edges": g["edges"],
        "speed_mean": float(np.mean(speeds)),
        "speed_max": float(np.max(speeds)),
        "trans": trans,
        "factor_med": float(np.median(f)) if f is not None and f.size else 0.0,
        "factor_max": float(f.max()) if f is not None and f.size else 0.0,
        "ratio_med": float(np.median(rr)) if rr.size else 0.0,
        "ratio_max": float(rr.max()) if rr.size else 0.0,
    }


print("comparing ceiling=None against several fixed caps")
print("{:>10}{:>6}{:>7}{:>10}{:>10}{:>7}{:>10}{:>10}{:>9}{:>9}".format(
    "ceiling", "att", "edges", "speed_avg", "speed_max", "trans",
    "f_med", "f_max", "r_med", "r_max"))
for c in (None, 8.0, 4.0, 2.0):
    r = run(c)
    label = "none" if c is None else "{:.1f}".format(c)
    print("{:>10}{:>6}{:>7}{:>10.4f}{:>10.4f}{:>7}{:>10.3f}{:>10.3f}{:>9.2f}{:>9.2f}".format(
        label, r["att"], r["edges"], r["speed_mean"], r["speed_max"],
        r["trans"], r["factor_med"], r["factor_max"],
        r["ratio_med"], r["ratio_max"]))
