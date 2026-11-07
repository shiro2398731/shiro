"""验证 P2 的端到端链路：上游信号 → L5 驱动量的映射。不启动进程。"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.runtime.process_2_mind import L5Runner
from src.L5_cognition.stream import Stream

D = 128
runner = L5Runner(dim=D, embed_dim=16, seed=1234)
stream = Stream(dim=D, gamma=0.05, dt=0.01, max_traces=64)

rng = np.random.default_rng(7)

# 1. 数据不足时不应该开始
for i in range(10):
    runner.surprise_hist.append(float(rng.random() * 0.1))
    runner.feature_hist.append(rng.random(16))
assert not runner.maybe_calibrate(stream), "样本不足时不应该校准"
print("1. 样本不足时不校准 OK")

# 2. 攒够数据后校准
for i in range(100):
    runner.surprise_hist.append(float(rng.random() * 0.1))
    runner.feature_hist.append(rng.random(16))

assert runner.maybe_calibrate(stream), "样本足够时应该校准"
assert stream.started, "校准后 stream 应该已启动"
print(f"2. 校准 OK，零点 norm={np.linalg.norm(stream.landscape.zero):.4f}")

# 3. 噪声幅度随惊讶变化
runner.surprise_hist.clear()
for v in [0.01] * 100:
    runner.surprise_hist.append(v)
a_flat = runner.noise_amplitude()
runner.surprise_hist.clear()
for v in [0.01 + 0.001 * (i % 5) for i in range(100)]:
    runner.surprise_hist.append(v)
a_flat2 = runner.noise_amplitude()
runner.surprise_hist.append(0.5)      # 突然一个大惊讶
a_spike = runner.noise_amplitude()
print(f"3. 平坦惊讶 noise={a_flat:.5f}；有微小波动时={a_flat2:.5f}；突增后={a_spike:.5f}")
assert a_spike > a_flat2, "惊讶突增时噪声幅度应增大"
assert a_flat == 0.0, "惊讶完全恒定时应为 0（无“惊讶的惊讶”）——此时靠惯性维持运动"
assert a_flat2 > 0 and a_flat2 < 0.1, "有微小波动时噪声应是小量"

# 4. 偏转场是 128 维
d = runner.deflection()
print(f"4. 偏转场 shape={d.shape if d is not None else None}")
assert d is not None and d.shape == (D,)

# 5. 驱动 L5 跑一段，确认没有报错且地形在长
stream.landscape.set_noise_amplitude(runner.noise_amplitude())
stream.landscape.set_deflection(d)
n0 = len(stream.state_space)
for i in range(400):
    if i % 50 == 0:
        runner.surprise_hist.append(float(rng.random() * 0.05))
        runner.feature_hist.append(rng.random(16))
        stream.landscape.set_noise_amplitude(runner.noise_amplitude())
        stream.landscape.set_deflection(runner.deflection())
    stream.step()
ss = stream.stream_state()
print(f"5. 跑了 400 步：attractors {n0}→{ss["n_attractors"]}  salient={ss["n_salient"]}  speed={ss['speed']:.3f}")
assert ss["n_attractors"] > n0, "地形应该有痕迹形成"

print("\nP2 链路全部通过")
