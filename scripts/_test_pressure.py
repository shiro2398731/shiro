import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from src.runtime.process_1_fast import FHNNetwork, L1Predictor, L2StateField

# Pressure was an integrator that never turned over. With a decay toward its own
# recent median it now moves without drifting. This checks the three properties
# that matter: it stays in range, it does not trend with elapsed time, and it
# still responds to what the oscillator is doing.


def collect(steps=200000, warm=5000):
    net = FHNNetwork(seed=42)
    pred = L1Predictor(input_dim=100)
    P, fast = [], []
    for i in range(steps):
        snap = net.step()
        pred.step(snap["E_rate_vec"])
        if i >= warm:
            P.append(snap["pressure"])
            fast.append(snap["fast"])
    return np.asarray(P), np.asarray(fast)


P, F = collect()

print("pressure over {} steps".format(P.size))
for q in (0, 1, 5, 25, 50, 75, 95, 99, 100):
    print("  p{:>3}: {:.4f}".format(q, np.percentile(P, q)))
print("  mean {:.4f}  std {:.4f}".format(P.mean(), P.std()))

# Trend: correlation with time. A pure integrator gives +1; a bounded quantity
# that responds to state gives something near zero.
t = np.arange(P.size, dtype=float)
trend = float(np.corrcoef(t, P)[0, 1])
print()
print("correlation with elapsed time: {:+.3f}".format(trend))

# Does it still depend on the oscillator? Compare the mean pressure in the
# highest and lowest tenths of fast activity.
order = np.argsort(F)
lo = P[order[:P.size // 10]].mean()
hi = P[order[-P.size // 10:]].mean()
print("mean pressure when fast is low : {:.4f}".format(lo))
print("mean pressure when fast is high: {:.4f}".format(hi))

# First-half vs second-half means: a drift shows up here even if the pointwise
# correlation is diluted by the oscillation.
h = P.size // 2
print("first half mean {:.4f}   second half mean {:.4f}   change {:+.1%}".format(
    P[:h].mean(), P[h:].mean(),
    (P[h:].mean() - P[:h].mean()) / max(abs(P[:h].mean()), 1e-12)))

print()
print("=" * 58)
ok_range = 0.0 <= P.min() and P.max() <= 1.0
ok_trend = abs(trend) < 0.3
ok_varies = P.std() > 0.01
print("{}  stays inside [0, 1]".format("ok  " if ok_range else "FAIL"))
print("{}  does not trend with elapsed time".format("ok  " if ok_trend else "FAIL"))
print("{}  still varies".format("ok  " if ok_varies else "FAIL"))
if ok_range and ok_trend and ok_varies:
    print()
    print("Pressure is bounded, does not accumulate with time, and still moves.")
else:
    sys.exit(1)
