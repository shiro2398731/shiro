"""L2 状态场验证：确认 dev 有真实波动、维度不贴边、归一化自适应。

顺序必须与主循环一致：先 field.step（用旧滑窗），再 feed_cycle（新周期入窗）。
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.runtime.process_1_fast import FHNNetwork, EnvelopeDetector, L2StateField, L1Predictor

net = FHNNetwork()
det = EnvelopeDetector()
pred = L1Predictor()
field = L2StateField()

t = 0.0
DT = 0.001
devy, vecs = [], []
last_fed = None

for i in range(150000):
    snap = net.step()
    t += DT
    s_t = pred.step(snap['E_rate_vec'])
    ph, per = det.update(snap['fast'], t)
    cs = det.last_cycle_stats

    v, dev = field.step(
        s_t=s_t, fast=snap['fast'], I=snap['I_rate'],
        mid=snap['mid'], slow=snap['slow'],
        pressure=snap['pressure'], cycle_stats=cs,
    )
    if cs is not None and cs is not last_fed:
        field.feed_cycle(cs)
        last_fed = cs

    if i > 80000:
        devy.append(dev)
        vecs.append(v)

devy = np.array(devy); vecs = np.array(vecs)
print(f"dev: mean={devy.mean():.4f} std={devy.std():.4f} min={devy.min():.4f} max={devy.max():.4f}")
print(f"dev 变化范围 = {devy.max()-devy.min():.4f}")
print()
print(f"{'dim':>3} {'mean':>7} {'std':>7} {'贴边率':>8}")
bad = []
for d in range(16):
    col = vecs[:, d]
    at_edge = np.mean((col <= 0.02) | (col >= 0.98))
    flag = ""
    if col.std() < 0.02:
        flag = "  <== 无变化"
        bad.append(d)
    if at_edge > 0.95:
        flag = "  <== 贴边"
        bad.append(d)
    print(f"{d:>3} {col.mean():>7.3f} {col.std():>7.3f} {at_edge:>7.1%}{flag}")
print(f"\n问题维度: {sorted(set(bad)) if bad else '无'}")
