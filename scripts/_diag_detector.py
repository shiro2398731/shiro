"""诊断：检测器在真实 fast 序列上的表现 vs 原始上穿间隔"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from src.runtime.process_1_fast import FHNNetwork, EnvelopeDetector

net = FHNNetwork()
det = EnvelopeDetector()

N = 120000   # 约 2 分钟墙钟（主循环每步 sleep 0.001）
fasts = np.empty(N)
t0 = time.time()
for i in range(N):
    snap = net.step()
    fasts[i] = snap['fast']
    det.update(snap['fast'], time.time())

# 原始真值：fast 相对全局中位数的上穿间隔
ref = np.median(fasts)
above = fasts >= ref
cross = np.where(above[1:] & ~above[:-1])[0]
gaps_raw = np.diff(cross) * 0.001     # 秒
print(f"原始上穿次数={len(cross)}")
print(f"原始周期: 中位数={np.median(gaps_raw):.4f}s min={gaps_raw.min():.4f} max={gaps_raw.max():.4f}")

print(f"\n检测器记录的 gaps ({len(det._gaps)} 个):")
for g in det._gaps:
    print(f"  {g:.4f}s")
print(f"检测器中位数: {det._period()}")

print(f"\nfast 分位数 5/50/95: {np.percentile(fasts, [5,50,95]).round(3)}")
print(f"elapsed {time.time()-t0:.1f}s")
