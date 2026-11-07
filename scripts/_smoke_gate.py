"""验证自适应更新：L2 只在该更新时更新，且更新率随活跃度变化。"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.runtime.process_1_fast import (
    FHNNetwork, EnvelopeDetector, L2StateField, L1Predictor, StalenessGate,
)

net = FHNNetwork()
det = EnvelopeDetector()
pred = L1Predictor()
field = L2StateField()
gate = StalenessGate(threshold=1.0)

t = 0.0
DT = 0.001
vec, dev = None, 0.0
updates = 0
steps = 0
last_fed = None
devy, stalenesses, update_flags = [], [], []

for i in range(120000):
    snap = net.step()
    t += DT
    steps += 1
    s_t = pred.step(snap['E_rate_vec'])
    ph, per = det.update(snap['fast'], t)
    cs = det.last_cycle_stats

    inputs = {
        "s_t": s_t, "fast": snap['fast'], "I": snap['I_rate'],
        "mid": snap['mid'], "slow": snap['slow'], "pressure": snap['pressure'],
    }
    s = gate.staleness(inputs)
    did = s > gate.threshold or vec is None
    if did:
        vec, dev = field.step(
            s_t=s_t, fast=snap['fast'], I=snap['I_rate'],
            mid=snap['mid'], slow=snap['slow'],
            pressure=snap['pressure'], cycle_stats=cs,
        )
        gate.commit(inputs)
        updates += 1
    if cs is not None and cs is not last_fed:
        field.feed_cycle(cs)
        last_fed = cs

    if i > 60000:
        devy.append(dev)
        stalenesses.append(s)
        update_flags.append(did)

print(f"L2 更新 {updates}/{steps} 步 = {updates/steps:.2%}")
print(f"后半段更新比例 = {np.mean(update_flags):.2%}")
print(f"dev: mean={np.mean(devy):.4f} std={np.std(devy):.4f} range={np.max(devy)-np.min(devy):.4f}")
print(f"staleness: mean={np.mean(stalenesses):.3f} p50={np.percentile(stalenesses,50):.3f} "
      f"p95={np.percentile(stalenesses,95):.3f} max={np.max(stalenesses):.2f}")

# 更新后的 dev 序列（只在更新时刻的 dev 才有意义）
print(f"\n更新时刻的 dev 数 = {len(field._vec_hist)}（协方差样本数）")
