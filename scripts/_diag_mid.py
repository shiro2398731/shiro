"""诊断 mid / slow 层饱和：看它们到底为什么二值跳变"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.runtime.process_1_fast import FHNNetwork

net = FHNNetwork()
mids, slows, drives_mid, drives_slow = [], [], [], []

for i in range(100000):
    snap = net.step()
    if i > 20000:
        mids.append(snap['mid'])
        slows.append(snap['slow'])

mids = np.array(mids); slows = np.array(slows)

def describe(name, a):
    print(f"{name}: mean={a.mean():.4f} std={a.std():.4f}")
    print(f"  分位 1/25/50/75/99: {np.percentile(a,[1,25,50,75,99]).round(4)}")
    print(f"  <0.001 比例: {np.mean(a<0.001):.1%}   >0.999 比例: {np.mean(a>0.999):.1%}")
    print(f"  取值唯一数(round 2): {len(np.unique(a.round(2)))}")

describe("mid ", mids)
describe("slow", slows)

# 看 mid 的驱动强度
E_rate = None
net2 = FHNNetwork()
drv_mid = []
for i in range(20000):
    snap = net2.step()
    Er = snap['E_rate_vec']
    d = net2.E_to_mid.T @ Er
    drv_mid.append(float(d.mean()))
drv_mid = np.array(drv_mid)
print(f"\nE_to_mid 驱动: mean={drv_mid.mean():.4f} max={drv_mid.max():.4f}")
print(f"  mid_I_base={net2.mid_I_base}, 求和后的 I 范围约 [{net2.mid_I_base}, {net2.mid_I_base+drv_mid.max():.2f}]")
print(f"  FHN 自激阈值 I_c ≈ {1 - 2/3*0.7**1.5:.4f} 左右（与 a,b 相关）")
