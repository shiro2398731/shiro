import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import collections
import numpy as np
from src.runtime.process_1_fast import (
    FHNNetwork, L1Predictor, L2StateField, EnvelopeDetector, StalenessGate,
)

# The gate threshold is in units of each channel's own typical fluctuation, so
# 1.0 has a meaning rather than being arbitrary: update when the input has moved
# by more than its own usual wobble. What is worth checking is what happens on
# either side of it, and whether 1.0 is on a plateau or on a cliff.


def run(threshold, steps=20000):
    net = FHNNetwork(seed=42)
    pred = L1Predictor(input_dim=100)
    field = L2StateField()
    gate = StalenessGate(threshold=threshold)
    env = EnvelopeDetector()
    vecs = []
    updates = 0
    last_fed = None
    for i in range(steps):
        snap = net.step()
        s_t = pred.step(snap["E_rate_vec"])
        cs = env.last_cycle_stats
        gi = {"s_t": s_t, "fast": snap["fast"], "I": snap["I_rate"],
              "mid": snap["mid"], "slow": snap["slow"],
              "pressure": snap["pressure"]}
        if gate.should_update(gi) or not vecs:
            v, _ = field.step(s_t=s_t, fast=snap["fast"], I=snap["I_rate"],
                              mid=snap["mid"], slow=snap["slow"],
                              pressure=snap["pressure"], cycle_stats=cs)
            gate.commit(gi)
            updates += 1
            if i > 4000:
                vecs.append(np.asarray(v, dtype=float))
        if cs is not None and cs is not last_fed:
            field.feed_cycle(cs)
            last_fed = cs
    V = np.stack(vecs) if vecs else np.zeros((0, 16))
    dev = V.std(axis=0).mean() if V.size else 0.0
    moving = int((V.std(axis=0) > 1e-4).sum()) if V.size else 0
    return updates / steps, dev, moving


print("{:>10}{:>12}{:>14}{:>12}".format("threshold", "update rate", "mean dim std", "moving dims"))
for t in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0):
    rate, dev, moving = run(t)
    print("{:>10.2f}{:>12.1%}{:>14.4f}{:>12}".format(t, rate, dev, moving))

print()
print("A threshold near 1.0 should sit where the update rate is neither 100%")
print("(every step, no gating at all) nor near zero (the field freezes).")
