"""诊断：为什么速度 runaway、地形参数实际是多少"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.L5_cognition.stream import Stream

D = 128
stream = Stream(dim=D, gamma=0.05, dt=0.01, recover_rate=0.002, max_traces=128)
rng = np.random.default_rng(42)
stream.calibrate(rng.normal(0.5, 0.1, size=(200, D)))
stream.landscape.set_noise_amplitude(0.02)

for i in range(800):
    stream.step()
    if i % 200 == 0 and i > 0:
        trs = stream.state_space.traces()
        sig = np.array([t.sigma for t in trs])
        amp = np.array([t.amp for t in trs])
        x = stream._st.x
        g = stream.landscape.gradient(x)
        pot = stream.landscape.potential(x)
        # 到最近痕迹的距离
        dmin = min(float(np.linalg.norm(x - t.mu)) for t in trs) if trs else -1
        print(f"step {i}: n_attractors={len(trs)}")
        print(f"   sigma: median={np.median(sig):.4f} max={sig.max():.4f}")
        print(f"   amp  : median={np.median(amp):.4f} max={amp.max():.4f}")
        print(f"   |x|={np.linalg.norm(x):.3f}  dmin={dmin:.3f}")
        print(f"   |grad|={np.linalg.norm(g):.4f}  pot={pot:.4f}")
        print(f"   |v|={np.linalg.norm(stream._st.v):.4f}")
