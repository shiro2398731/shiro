"""
scripts/_prof_step.py

Where does a step's time go?

Breaks one step down into its parts and reports each, on the real checkpoint.
Uses cProfile plus manual timing so the numbers can be checked against each
other.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_prof_step.py
"""

import cProfile
import io
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.L5_cognition.stream import Stream
import src.common.persist as persist

s = Stream(dim=128)
if not persist.load_l5(s):
    print("no checkpoint")
    sys.exit(1)

for _ in range(100):
    s.landscape.set_noise_amplitude(0.015)
    s.step()

n_attr = len(s.state_space)
n_edges = len(s.graph)
print(f"attractors={n_attr}  edges={n_edges}")

# ---- manual breakdown ----
x = s._st.x.copy()
sp = s.state_space

t0 = time.perf_counter()
for _ in range(200):
    sp.potential(x)
t_pot = (time.perf_counter() - t0) / 200

t0 = time.perf_counter()
for _ in range(200):
    sp.gradient(x)
t_grad = (time.perf_counter() - t0) / 200

t0 = time.perf_counter()
for _ in range(200):
    sp.salient()
t_sal = (time.perf_counter() - t0) / 200

t0 = time.perf_counter()
for _ in range(200):
    sp.current_scale()
t_scale = (time.perf_counter() - t0) / 200

t0 = time.perf_counter()
for _ in range(200):
    s.drive.tensions(sp, s.graph)
t_ten = (time.perf_counter() - t0) / 200

t0 = time.perf_counter()
for _ in range(200):
    s.recaller.by_cue(sp, s.graph, x)
t_rec = (time.perf_counter() - t0) / 200

print()
print("per-call cost:")
print(f"  potential      {t_pot*1000:8.3f} ms")
print(f"  gradient       {t_grad*1000:8.3f} ms")
print(f"  salient        {t_sal*1000:8.3f} ms")
print(f"  current_scale  {t_scale*1000:8.3f} ms")
print(f"  tensions       {t_ten*1000:8.3f} ms")
print(f"  recall.by_cue  {t_rec*1000:8.3f} ms")

# ---- whole step ----
t0 = time.perf_counter()
N = 200
for _ in range(N):
    s.landscape.set_noise_amplitude(0.015)
    s.step()
t_step = (time.perf_counter() - t0) / N
print(f"\n  whole step     {t_step*1000:8.3f} ms")

# ---- profile ----
pr = cProfile.Profile()
pr.enable()
for _ in range(100):
    s.landscape.set_noise_amplitude(0.015)
    s.step()
pr.disable()

buf = io.StringIO()
ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
ps.print_stats(18)
print("\n--- cProfile (cumulative) ---")
print(buf.getvalue()[:3000])
