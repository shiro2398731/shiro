"""
scripts/_diag_metabolism.py

Measure the balance between deposit and recovery in the L5 terrain.

Question this answers:
    In real operation, does a trace's depth grow faster than it fades?
    If not, new structure can never take hold.

How it measures (all from a live Stream, no invented numbers):
    1. Deposit gain  : mean depth increase caused by one pass_through
    2. Recovery loss : mean depth lost by one recover() call
    3. Half-life     : how many recovery calls for depth to halve
    4. Live run      : how often new traces appear vs how many survive
    5. Depth profile : the actual shape of her terrain right now

It runs a synthetic driver for N steps, but reads the REAL parameters
from the Stream, so the numbers reflect production behaviour.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_diag_metabolism.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream
from src.L5_cognition.terrain import Terrain

D = 128
N = 3000

s = Stream(dim=D, gamma=0.05, dt=0.01, seed=7)
warm = np.cumsum(np.random.default_rng(1).normal(0, 0.05, (64, D)), axis=0)
s.calibrate(warm)

t = s.state_space
print("--- terrain parameters ---")
print(f"recover_rate   : {t.recover_rate}")
print(f"deposit_ratio  : {t.deposit_ratio}")
print(f"ceiling        : {t.ceiling}")
print(f"min_amp        : {t.min_amp}")
print(f"recover_every  : {s._recover_every} steps")

# ---------- probe: one isolated trace ----------

rng = np.random.default_rng(5)
probe_t = Terrain(dim=D, recover_rate=t.recover_rate,
                  deposit_ratio=t.deposit_ratio, ceiling=t.ceiling,
                  min_amp=t.min_amp)
x0 = rng.normal(0, 0.1, D)
for _ in range(120):
    probe_t.pass_through(x0 + rng.normal(0, 0.01, D))

if len(probe_t) == 0:
    print("no trace formed in probe; abort")
    sys.exit(0)

print(f"\n--- probe terrain after 120 passes ---")
print(f"traces in probe: {len(probe_t)}")

if len(probe_t) > 0:
    probe = max(probe_t.traces(), key=lambda z: z.amp)

    amp_before = probe.amp
    probe_t.pass_through(x0 + rng.normal(0, 0.01, D))
    amp_after = probe.amp
    gain = amp_after - amp_before

    amp_p = probe.amp
    probe_t.recover(rng)
    amp_q = probe.amp
    loss = amp_p - amp_q
    loss_pct = loss / amp_p if amp_p > 0 else 0.0

    print(f"depth now          : {amp_p:.6f}  (ceiling {t.ceiling:.3f})")
    print(f"gain per pass      : {gain:+.6f}")
    print(f"loss per recover   : {loss:.6f}  ({loss_pct:.3%})")

    if loss_pct > 0:
        halflife = np.log(0.5) / np.log(max(1.0 - loss_pct, 1e-12))
        print(f"half-life          : {halflife:.1f} recover calls "
              f"= {halflife * s._recover_every:.0f} steps")

    # break-even: how many passes needed per recovery period to hold depth
    passes_needed = loss / gain if gain > 0 else float("inf")
    print(f"passes needed per {s._recover_every} steps to hold depth: "
          f"{passes_needed:.2f}")

# ---------- live stream ----------

print("\n--- live stream ---")
births = 0
prev_n = len(s.state_space)
drift = np.zeros(D)
drng = np.random.default_rng(99)

for i in range(N):
    drift += drng.normal(0, 0.02, D) * 0.3
    s.landscape.set_noise_amplitude(0.02)
    s.landscape.set_deflection(drift)
    s.step()
    n = len(s.state_space)
    if n > prev_n:
        births += 1
    prev_n = n

alive = len(s.state_space)
print(f"steps              : {N}")
print(f"traces now         : {alive}")
print(f"steps that created : {births}  ({births / N:.1%} of steps)")
print(f"edges              : {len(s.graph)}")
print(f"salient             : {len(s.state_space.salient())}")

amps = np.array([tr.amp for tr in s.state_space.traces()])
if len(amps):
    print(f"depth p10/p50/p90  : {np.percentile(amps,10):.5f} / "
          f"{np.percentile(amps,50):.5f} / {np.percentile(amps,90):.5f}")
    print(f"depth max          : {amps.max():.5f}")
    print(f"ceiling / max      : {t.ceiling / amps.max():.1f}x")
