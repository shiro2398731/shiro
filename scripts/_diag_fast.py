"""诊断 L0 fast 的真实时间尺度：它在原始步长上到底多快翻转。"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from src.runtime.process_1_fast import FHNNetwork

net = FHNNetwork()
N = 200000          # 200k 步，dt=0.1，约 2 万步/秒
vals = np.empty(N)
for i in range(N):
    vals[i] = net.step()['fast']

ref = np.median(vals)
above = vals >= ref
cross = np.where((above[1:]) & (~above[:-1]))[0]
if len(cross) > 1:
    gaps = np.diff(cross)
    print(f"中位数={ref:.4f}")
    print(f"上穿次数={len(cross)}")
    print(f"上穿间隔(步): 中位数={np.median(gaps):.1f} 均值={gaps.mean():.1f} min={gaps.min()} max={gaps.max()}")
    print(f"=> 若每步 0.1 时间单位，周期约 {np.median(gaps)*0.1:.1f} 时间单位")
print(f"fast 前 40 个原始值: {np.round(vals[:40], 3)}")
print(f"fast 分位数 5/50/95: {np.percentile(vals, [5,50,95]).round(3)}")
