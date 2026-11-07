"""深挖：检测器内部为什么几乎不记录上穿"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.runtime.process_1_fast import FHNNetwork, EnvelopeDetector

net = FHNNetwork()
det = EnvelopeDetector()

print(f"SAMPLE_INTERVAL_S = {det.SAMPLE_INTERVAL_S}")
print(f"HIST_WINDOW_S = {det.HIST_WINDOW_S}")

t = 0.0
DT = 0.001
flips = 0
prev_above = None
for i in range(40000):
    snap = net.step()
    t += DT
    ph, per = det.update(snap['fast'], t)
    # 观察检测器内部 above 状态
    if prev_above is not None and det._was_above != prev_above:
        flips += 1
    prev_above = det._was_above

print(f"\n检测器 above 状态翻转次数 = {flips} (原始应有约 {int(20/0.347)} 次)")
print(f"hist 长度 = {len(det._hist_v)}")
print(f"gaps = {list(det._gaps)}")
if len(det._hist_v) > 0:
    hv = np.array(det._hist_v)
    print(f"hist 分位数 5/50/95 = {np.percentile(hv,[5,50,95]).round(3)}")
    print(f"hist 时间跨度 = {det._hist_t[-1] - det._hist_t[0]:.2f}s")
