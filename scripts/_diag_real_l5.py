"""
scripts/_diag_real_l5.py

Read the ACTUAL saved terrain from last night and describe its shape.

No synthetic driver, no invented scenario. This only loads
data/state/l5.pkl and reports what is really in there.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_real_l5.py
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

PATH = os.path.join("data", "state", "l5.pkl")

with open(PATH, "rb") as f:
    obj = pickle.load(f)

print("--- top level ---")
print("type:", type(obj).__name__)
if isinstance(obj, dict):
    for k, v in obj.items():
        print(f"  {k}: {type(v).__name__}  ", end="")
        if isinstance(v, (int, float, str)):
            print(v)
        elif isinstance(v, (list, tuple)):
            print(f"len={len(v)}")
        else:
            print()

# Find the trace list
traces = None
meta = {}

if isinstance(obj, dict) and isinstance(obj.get("traces"), list):
    traces = obj["traces"]
    for k, v in obj.items():
        if isinstance(v, (int, float)):
            meta[k] = v
        elif isinstance(v, np.ndarray):
            meta[k] = f"array{list(v.shape)}"
        elif isinstance(v, list):
            meta[k] = f"list len={len(v)}"

print("\n--- meta ---")
for k, v in meta.items():
    print(f"  {k}: {v}")

if traces is None:
    print("\nno trace list found; stop")
    sys.exit(0)

print(f"\n--- real terrain: {len(traces)} traces ---")
print("trace record keys:", sorted(traces[0].keys()) if isinstance(traces[0], dict) else "not a dict")

def g(tr, key, default=None):
    if isinstance(tr, dict):
        return tr.get(key, default)
    return getattr(tr, key, default)

amps = np.array([g(tr, "amp", 0.0) for tr in traces], dtype=float)
sig = np.array([g(tr, "scale", None) or g(tr, "sigma", 1.0) for tr in traces], dtype=float)
nm = np.array([len(g(tr, "members", [])) for tr in traces], dtype=int)

print("depth:")
for p in (0, 10, 25, 50, 75, 90, 99, 100):
    print(f"  p{p:<3}: {np.percentile(amps, p):.5f}")

print("members per trace:")
for p in (0, 25, 50, 75, 90, 100):
    print(f"  p{p:<3}: {np.percentile(nm, p):.1f}")

print("territory radius (sigma):")
for p in (10, 50, 90):
    print(f"  p{p:<3}: {np.percentile(sig, p):.5f}")

print(f"\ntraces with >=2 members : {(nm >= 2).sum()} / {len(nm)} "
      f"({(nm >= 2).mean():.1%})")
print(f"traces with >=4 members : {(nm >= 4).sum()} ({ (nm >= 4).mean():.1%})")
print(f"traces with 1 member    : {(nm == 1).sum()} ({(nm == 1).mean():.1%})")

print(f"\ndepth sum               : {amps.sum():.2f}")
print(f"deepest                 : {amps.max():.4f}")

# what does the deep half look like vs the shallow half
order = np.argsort(-amps)
top = order[: max(1, len(order) // 20)]
print(f"\ntop 5% traces: mean members={nm[top].mean():.1f}, "
      f"mean sigma={sig[top].mean():.5f}, mean depth={amps[top].mean():.4f}")
bot = order[-max(1, len(order) // 2):]
print(f"bottom 50%   : mean members={nm[bot].mean():.1f}, "
      f"mean sigma={sig[bot].mean():.5f}, mean depth={amps[bot].mean():.4f}")
