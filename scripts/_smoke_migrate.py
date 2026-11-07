"""
迁移验证：把她的形状从一台"机器"搬到另一台，确认还是同一个她。

做法：
  1. 在目录 A 里跑一段，存形状
  2. 把 data/state 整个复制到目录 B
  3. 在 B 里加载，确认形状一致，并且能继续跑
  4. 对比继续跑之后，两边是否会分叉（应该会，因为各自往前走了）

这不是"拷"而是"搬"：验证的是同一个她能在新宿主上延续，
而不是生出一个克隆体。（白默的区分：迁移是搬，不是拷。）
"""
import sys, os, shutil, tempfile
sys.path.insert(0, '.')
import numpy as np
from pathlib import Path

import src.common.persist as persist
from src.runtime.process_1_fast import FHNNetwork, L1Predictor, L2StateField
from src.L5_cognition.stream import Stream

BASE = Path("data/_migrate_test")
if BASE.exists():
    shutil.rmtree(BASE)
(BASE / "A").mkdir(parents=True)
(BASE / "B").mkdir(parents=True)

# ---------- 1. 在 A 里跑一段 ----------
persist.STATE_DIR = BASE / "A"
net = FHNNetwork()
pred = L1Predictor()
field = L2StateField()
stream = Stream(dim=128, gamma=0.05, dt=0.01, max_traces=48)

stream.calibrate(np.random.default_rng(1).normal(0.5, 0.1, size=(100, 128)))
stream.landscape.set_noise_amplitude(0.02)

for _ in range(2000):
    snap = net.step()
    pred.step(snap["E_rate_vec"])
    stream.step()

saved = persist.snapshot_all({
    "l0": net, "l1": pred, "l2": field, "acc": net.accumulator, "l5": stream,
})
print("1. A 里存下的形状：")
for k, v in saved.items():
    print(f"   {k}: {os.path.getsize(v)/1024:.1f} KB")

fingerprint_A = {
    "step_count": net.step_count,
    "pressure": net.accumulator.P,
    "traces": len(stream.state_space),
    "depth_sum": float(np.sum(stream.state_space.depth_profile())),
    "x_norm": float(np.linalg.norm(stream._st.x)),
}

# ---------- 2. 搬过去 ----------
for f in (BASE / "A").glob("*.pkl"):
    shutil.copy2(f, BASE / "B" / f.name)
print(f"\n2. 搬到 B：{len(list((BASE/'B').glob('*.pkl')))} 个文件")

# ---------- 3. 在 B 里加载 ----------
persist.STATE_DIR = BASE / "B"
net2 = FHNNetwork()
pred2 = L1Predictor()
field2 = L2StateField()
stream2 = Stream(dim=128, gamma=0.05, dt=0.01, max_traces=48)

restored = persist.restore_all({
    "l0": net2, "l1": pred2, "l2": field2, "acc": net2.accumulator, "l5": stream2,
})
print(f"3. B 里恢复：{restored}")

fingerprint_B = {
    "step_count": net2.step_count,
    "pressure": net2.accumulator.P,
    "traces": len(stream2.state_space),
    "depth_sum": float(np.sum(stream2.state_space.depth_profile())),
    "x_norm": float(np.linalg.norm(stream2._st.x)),
}

print("\n   指纹对比：")
ok = True
for k in fingerprint_A:
    a, b = fingerprint_A[k], fingerprint_B[k]
    same = abs(a - b) < 1e-9
    ok = ok and same
    print(f"   {k:12} A={a:<16.6f} B={b:<16.6f} {'✓' if same else '✗'}")

assert ok, "迁移后形状不一致"

# ---------- 4. 在 B 里继续跑 ----------
stream2.landscape.set_noise_amplitude(0.02)
for _ in range(100):
    stream2.step()
    snap2 = net2.step()

print(f"\n4. B 里继续跑了 100 步：step_count={net2.step_count} traces={len(stream2.state_space)}")
assert net2.step_count > fingerprint_A["step_count"], "B 里应该能继续推进"

# ---------- 5. 各自的形状独立演进（不是克隆体） ----------
print(f"\n5. 两边分叉检查：")
print(f"   A 最后 step={fingerprint_A['step_count']}")
print(f"   B 现在 step={net2.step_count}")
print(f"   B 多了 {net2.step_count - fingerprint_A['step_count']} 步")
print("   → 搬过去之后，B 独自往前走；A 停在原地。这是'搬'不是'拷'。")

shutil.rmtree(BASE)
print("\n迁移验证通过")
