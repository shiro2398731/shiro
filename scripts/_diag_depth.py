"""
scripts/_diag_depth.py

Why are most attractors shallow?

Reads the real checkpoint and reports one thing only: the relation between
how often a region was observed and how deep it became.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_depth.py
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

PATH = os.path.join("data", "state", "l5.pkl")

with open(PATH, "rb") as f:
    obj = pickle.load(f)

raw = obj.get("attractors") or obj.get("traces") or []
step_count = obj["step_count"]
gscale = obj.get("global_scale")

print(f"step_count   : {step_count}")
print(f"attractors   : {len(raw)}")
print(f"global_scale : {gscale}")

nm = np.array([len(r["members"]) for r in raw], dtype=float)
amps = np.array([r["amp"] for r in raw], dtype=float)
scale = np.array([r.get("scale") or 0.0 for r in raw], dtype=float)

order = np.argsort(-nm)
nm, amps, scale = nm[order], amps[order], scale[order]

mx = amps.max() if len(amps) and amps.max() > 0 else 1.0

print("\n--- sorted by member count ---")
print(f"{'members':>9} {'amp':>10} {'amp/max':>8} {'scale':>10}")
idx = list(range(min(8, len(nm)))) + list(range(max(0, len(nm) - 4), len(nm)))
for i in idx:
    print(f"{int(nm[i]):>9} {amps[i]:>10.5f} {amps[i]/mx:>8.3f} {scale[i]:>10.6f}")

print("\n--- grouped by member count ---")
for lo, hi, lbl in [(1, 1, "1"), (2, 3, "2-3"), (4, 9, "4-9"),
                    (10, 99, "10-99"), (100, 9999, "100-9999"),
                    (10000, 10**12, "10000+")]:
    sel = (nm >= lo) & (nm <= hi)
    k = int(sel.sum())
    if k:
        print(f"  members {lbl:>9}: {k:>4} attractors   "
              f"mean amp {amps[sel].mean():.5f}   "
              f"max amp {amps[sel].max():.5f}")

print("\n--- distribution ---")
print(f"deepest : {amps.max():.4f}")
print(f"p99     : {np.percentile(amps, 99):.5f}")
print(f"p90     : {np.percentile(amps, 90):.5f}")
print(f"median  : {np.median(amps):.5f}")
print(f"amp < 1% of deepest : {int((amps < mx * 0.01).sum())} / {len(amps)}")
print(f"amp < 5% of deepest : {int((amps < mx * 0.05).sum())} / {len(amps)}")

print("\n--- pure accumulation reference (a = 0.05, no recovery) ---")
a = 0.05
for m in [1, 2, 4, 10, 50, 200, 1000]:
    print(f"  if assigned {m:>5} times -> amp would be {1 - (1 - a) ** m:.5f}")

print("\n--- observed vs pure accumulation, for the largest ones ---")
for i in range(min(6, len(nm))):
    m = int(nm[i])
    pred = 1 - (1 - a) ** m
    print(f"  members {m:>8}: observed {amps[i]:.5f}   "
          f"pure-accumulation {pred:.5f}   ratio {amps[i]/pred:.4f}")
