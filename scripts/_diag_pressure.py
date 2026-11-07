import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from src.runtime.process_1_fast import FHNNetwork

# Pressure is an integrator: it rises while the fast activity is above its own
# recent median and falls while it is below. Over a long run those two should
# roughly balance, so its level says something about asymmetry rather than about
# elapsed time.
#
# Two earlier observations disagreed: one run of 60000 steps reached 0.0024,
# another of 20000 showed a normalised value sitting between 0.69 and 1.0. Both
# cannot be right about the same quantity, so this watches the raw value.

net = FHNNetwork(seed=42)
vals = []
for i in range(400000):
    snap = net.step()
    if i % 20000 == 0 and i > 0:
        vals.append((i, snap["pressure"], snap["fast"], net.accumulator.P))

print("{:>8}{:>12}{:>10}{:>12}".format("step", "pressure", "fast", "P_raw"))
for v in vals:
    print("{:>8}{:>12.6f}{:>10.4f}{:>12.6f}".format(*v))

print()
print("final pressure", round(float(net.accumulator.P), 6))
print("tau_up", net.accumulator.tau_up, "tau_down", net.accumulator.tau_down)
print("dt", net.accumulator.dt)

# Fraction of time fast is above its own median determines which tau applies.
hist = []
above_count = 0
for i in range(20000):
    f = net.step()["fast"]
    hist.append(f)
    if len(hist) > 200:
        hist.pop(0)
    if len(hist) > 8 and f > float(np.median(hist)):
        above_count += 1
print("fraction of steps above own median: {:.3f}".format(above_count / 20000))
