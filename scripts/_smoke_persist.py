"""验证持久化：存形状 → 改它 → 读回来 → 确认恢复的是一模一样的形状。"""
import sys, os, shutil
sys.path.insert(0, '.')
import numpy as np

# 用临时目录，不污染真实状态
TMP = "data/_test_state"
if os.path.exists(TMP):
    shutil.rmtree(TMP)
os.makedirs(TMP)

import src.common.persist as persist
persist.STATE_DIR = __import__("pathlib").Path(TMP)

from src.runtime.process_1_fast import FHNNetwork, L1Predictor, L2StateField
from src.L5_cognition.stream import Stream

# ---------- L0 ----------
net = FHNNetwork()
for _ in range(500):
    snap = net.step()
E_v_before = net.E_v.copy()
P_before = net.accumulator.P
step_before = net.step_count

persist.save_l0(net)
persist.save_acc(net.accumulator)

# 打乱它
net.E_v[:] = 0.0
net.accumulator.P = 0.0
net.step_count = 0

ok = persist.load_l0(net)
ok2 = persist.load_acc(net.accumulator)
print(f"L0 恢复: {ok}, acc 恢复: {ok2}")
assert np.allclose(net.E_v, E_v_before), "E_v 没恢复"
assert abs(net.accumulator.P - P_before) < 1e-12, "pressure 没恢复"
assert net.step_count == step_before, "step_count 没恢复"
print(f"  step_count={net.step_count}  pressure={net.accumulator.P:.4f}  OK")

# ---------- L1 ----------
pred = L1Predictor()
for _ in range(100):
    pred.step(np.random.rand(100))
W_before = pred.W_pred.copy()
persist.save_l1(pred)
pred.W_pred[:] = 0.0
persist.load_l1(pred)
assert np.allclose(pred.W_pred, W_before), "L1 权重没恢复"
print(f"  L1 权重恢复 OK, norm={np.linalg.norm(pred.W_pred):.4f}")

# ---------- L2 ----------
field = L2StateField()
for i in range(200):
    field.step(s_t=0.05, fast=0.5, I=0.4, mid=0.3, slow=0.2,
               pressure=0.1, cycle_stats=None)
n_dim_hist = [len(h) for h in field._dim_hist]
persist.save_l2(field)
field2 = L2StateField()
persist.load_l2(field2)
assert [len(h) for h in field2._dim_hist] == n_dim_hist, "L2 分位历史没恢复"
print(f"  L2 分位历史恢复 OK, 各维样本数={n_dim_hist[:4]}...")

# ---------- L5 ----------
stream = Stream(dim=128, gamma=0.05, dt=0.01, max_traces=64)
stream.calibrate(np.random.normal(0.5, 0.1, size=(100, 128)))
stream.landscape.set_noise_amplitude(0.02)
for _ in range(600):
    stream.step()
n_attractors = len(stream.state_space)
depth_sum = float(np.sum(stream.state_space.depth_profile()))
x_before = stream._st.x.copy()

persist.save_l5(stream)

stream2 = Stream(dim=128, gamma=0.05, dt=0.01, max_traces=64)
ok5 = persist.load_l5(stream2)
print(f"L5 恢复: {ok5}")
assert len(stream2.state_space) == n_attractors, f"痕迹数不对: {len(stream2.state_space)} vs {n_attractors}"
assert abs(float(np.sum(stream2.state_space.depth_profile())) - depth_sum) < 1e-9, "深度总和不对"
assert np.allclose(stream2._st.x, x_before), "位置没恢复"
print(f"  痕迹数={len(stream2.state_space)}  深度总和={depth_sum:.3f}  OK")

# 恢复后能继续跑
for _ in range(10):
    stream2.step()
print(f"  恢复后继续跑了 10 步，痕迹={len(stream2.state_space)}")

# 检查存了什么
import glob
for f in sorted(glob.glob(TMP + "/*")):
    print(f"  {os.path.basename(f)}: {os.path.getsize(f)/1024:.1f} KB")

shutil.rmtree(TMP)
print("\n全部通过")
