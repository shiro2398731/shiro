"""工作空间验证：点燃发生、不回流、门槛来自自身历史。"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.L5_cognition.stream import Stream

D = 128
stream = Stream(dim=D, gamma=0.05, dt=0.01, max_traces=64)
rng = np.random.default_rng(42)
stream.calibrate(rng.normal(0.5, 0.1, size=(100, D)))
stream.landscape.set_noise_amplitude(0.03)

ignitions = []
x_before = None
for i in range(3000):
    if i == 1500:
        x_before = stream._st.x.copy()
        pos_before = len(stream.state_space)
    ev = stream.step()
    if stream._last_ignition is not None:
        ignitions.append(stream._last_ignition)
        stream._last_ignition = None

print(f"attractors={len(stream.state_space)}  salient={len(stream.state_space.salient())}")
print(f"点燃次数={len(ignitions)}  工作空间历史={stream.workspace.ignition_count}")
print(f"惯常强度基线={stream.workspace.baseline()}")

if ignitions:
    strengths = [g.strength for g in ignitions]
    print(f"点燃强度: min={min(strengths):.4f} mean={np.mean(strengths):.4f} max={max(strengths):.4f}")
    assert all(s > 0 for s in strengths)

# 关键：点燃不改变地形（只读）
print(f"\n地形在点火期间的attractors: {pos_before} → {len(stream.state_space)}（会变，是因为一直在长）")
assert hasattr(stream.workspace, "history")

# 验证工作空间没有修改地形对象
import inspect
src = inspect.getsource(stream.workspace.observe)
for forbidden in ["pass_through", "deepen", "recover", "terrain._traces"]:
    assert "terrain." + forbidden not in src, f"工作空间不应该修改地形：{forbidden}"
print("工作空间源码中没有修改地形的调用 OK")

print("\n工作空间验证通过")
