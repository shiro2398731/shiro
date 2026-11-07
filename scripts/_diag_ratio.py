import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import src.common.persist as persist
from src.L5_cognition.stream import Stream
from src.L4_drive.viability import Viability, _interval_scale

s = Stream(dim=128)
persist.load_l5(s)
ss = s.landscape.state_space
V = Viability()
a = ss.assign_ratio
C = ss.ceiling

rows = []
for A in ss.attractors():
    p = A._assigned / max(ss._total_steps, 1)
    car = V.carrying_depth(ss, A, p)
    if car > 0:
        rows.append((A._assigned, A.amp, car, A.amp / car, _interval_scale(ss, A)))
rows.sort()

hdr = "{:>7}{:>10}{:>10}{:>9}{:>9}".format("assign", "amp", "carry", "ratio", "isc")
print(hdr)
mid = len(rows) // 2
for r in rows[:5] + rows[mid - 2:mid + 2] + rows[-5:]:
    print("{:>7}{:>10.4f}{:>10.4f}{:>9.2f}{:>9.3f}".format(*r))

ratios = np.array([r[3] for r in rows])
amps = np.array([r[1] for r in rows])
asg = np.array([float(r[0]) for r in rows])
print()
print("corr(ratio, amp)      =", round(float(np.corrcoef(ratios, amps)[0, 1]), 3))
print("corr(ratio, assigned) =", round(float(np.corrcoef(ratios, asg)[0, 1]), 3))
print("assign_ratio =", a, " ceiling =", C)
