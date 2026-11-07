"""看 L5 的真实轨迹：位置尺度、步幅、停留结构"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.L5_cognition.stream import Stream

D = 128
stream = Stream(dim=D, gamma=0.05, dt=0.01, recover_rate=0.002, max_traces=256)
rng = np.random.default_rng(42)
stream.calibrate(rng.normal(0.5, 0.1, size=(200, D)))
stream.landscape.set_noise_amplitude(0.02)

pos = []
steps = []
for i in range(1500):
    x0 = stream._st.x.copy()
    stream.step()
    x1 = stream._st.x.copy()
    pos.append(x1)
    steps.append(float(np.linalg.norm(x1 - x0)))

pos = np.array(pos)
steps = np.array(steps)

print(f"步幅: mean={steps.mean():.5f} median={np.median(steps):.5f} "
      f"p95={np.percentile(steps,95):.5f} max={steps.max():.5f}")
print(f"总位移 = {np.linalg.norm(pos[-1]-pos[0]):.3f}")
print(f"位置各维 std 中位数 = {np.median(pos.std(axis=0)):.4f}")
print(f"位置范数: mean={np.linalg.norm(pos,axis=1).mean():.3f} "
      f"std={np.linalg.norm(pos,axis=1).std():.3f}")

# 最近 32 步的位置离散度（这就是我想拿来做 sigma 的量）
for N in (8, 16, 32, 64):
    sig = []
    for i in range(N, len(pos), 50):
        w = pos[i-N:i]
        sig.append(float(np.sqrt(((w - w.mean(axis=0))**2).sum(axis=1).mean())))
    sig = np.array(sig)
    print(f"最近 {N:>2} 步的 RMS 离散: median={np.median(sig):.5f} "
          f"p90={np.percentile(sig,90):.5f}")

prof = stream.state_space.depth_profile()
print(f"\n痕迹数={len(prof)} 深度 median={np.median(prof):.4f} max={prof.max():.4f}")
print(f"sigma 中位数 = {np.median([t.sigma for t in stream.state_space.traces()]):.5f}")
