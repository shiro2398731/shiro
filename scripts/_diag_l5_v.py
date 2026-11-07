"""逐步追踪 v 的来源"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.L5_cognition.stream import Stream

D = 128
stream = Stream(dim=D, gamma=0.05, dt=0.01, recover_rate=0.002, max_traces=64)
rng = np.random.default_rng(42)
stream.calibrate(rng.normal(0.5, 0.1, size=(200, D)))
stream.landscape.set_noise_amplitude(0.02)

print(f"{'step':>5} {'|v|':>10} {'|grad|':>10} {'|x|':>10} {'dv_noise':>10} {'dv_damp':>10} {'dv_grad':>10}")
for i in range(60):
    st_before = stream._st
    v_before = st_before.v.copy()
    g = stream.landscape.gradient(st_before.x)

    stream.step()
    v_after = stream._st.v

    dv = v_after - v_before
    dv_damp = -stream.dynamics.gamma * v_before * stream.dynamics.dt
    dv_grad = -g * stream.dynamics.dt
    dv_noise = dv - dv_damp - dv_grad

    if i % 10 == 0 or i < 5:
        print(f"{i:>5} {np.linalg.norm(v_after):>10.4f} {np.linalg.norm(g):>10.4f} "
              f"{np.linalg.norm(stream._st.x):>10.4f} "
              f"{np.linalg.norm(dv_noise):>10.5f} {np.linalg.norm(dv_damp):>10.5f} "
              f"{np.linalg.norm(dv_grad):>10.5f}")
