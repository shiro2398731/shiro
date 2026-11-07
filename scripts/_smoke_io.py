"""
scripts/_smoke_io.py

End-to-end check of the language interface, in one process.

Exercises the whole path that the four processes would exercise:

    text -> encoder -> TextSignalEvent -> FHNNetwork.receive -> L1 error

and reports what changed. No bus, no network: the point is to confirm the
arithmetic and the wiring, not the transport.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_smoke_io.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L6_expression.encoder import HashEncoder, make_encoder
from src.runtime.process_1_fast import FHNNetwork, L1Predictor

D = 100


def run(texts, steps_per_text=40, gain=0.15):
    net = FHNNetwork(seed=42)
    pred = L1Predictor(input_dim=D, proj_dim=8)
    enc = HashEncoder(dim=128)

    # warm up so the oscillator is in its own rhythm
    for _ in range(200):
        net.step()

    rows = []
    for t in texts:
        sig = enc.encode(t)
        net.receive(sig.vector, gain=gain)
        errs = []
        means = []
        for _ in range(steps_per_text):
            snap = net.step()
            s_t = pred.step(snap["E_rate_vec"])
            errs.append(s_t)
            means.append(snap["fast"])
        rows.append({
            "text": t,
            "mean_rate": float(np.mean(means)),
            "peak_error": float(np.max(errs)),
            "decayed": float(net._ext_amp),
            "tags": sig.tags,
        })
    return rows


enc_kind = make_encoder(dim=128)[1]
print(f"encoder available: {enc_kind}")
print()

rows = run(["今天好累", "今天很好", "你还记得我吗？", "今天好累"])
print(f"{'text':<16} {'mean':>8} {'peak_s':>9} {'ext_left':>9}")
for r in rows:
    print(f"{r['text']:<16} {r['mean_rate']:>8.4f} {r['peak_error']:>9.4f} "
          f"{r['decayed']:>9.5f}")

print()
print("checks")
base = rows[0]["mean_rate"]
same = rows[3]["mean_rate"]
print(f"  same sentence twice gives close means : "
      f"{abs(base - same) < 0.15}  ({base:.4f} vs {same:.4f})")
diff = rows[1]["mean_rate"]
print(f"  different sentences give different means : "
      f"{abs(base - diff) > 1e-4}  ({base:.4f} vs {diff:.4f})")
print(f"  drive decays after each signal : "
      f"{all(r['decayed'] < 0.15 for r in rows)}")
print(f"  question tag detected : {rows[2]['tags']['question']}")
