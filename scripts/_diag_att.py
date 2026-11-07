"""排查：为什么 depth=1 但 potential=0"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.L5_cognition.attractors import AttractorSet

rng = np.random.default_rng(42)
D, k = 128, 32
X = rng.normal(0.5, 0.1, size=(200, D))

s = AttractorSet(dim=D, subspace_dim=k)
a = s.grow(X)
s.deepen(a.attractor_id, 1.0)

print(f"depth={a.depth}")
print(f"origin 形状={a.origin.shape} 范数={np.linalg.norm(a.origin):.4f}")
print(f"basis 形状={a.basis.shape}")
print(f"center 形状={a.center.shape} 范数={np.linalg.norm(a.center):.4f}")
print(f"cov 迹={np.trace(a.cov):.6f}")

# 取一个应落在吸引子内的点：样本均值
x = X.mean(axis=0)
p = a.project(x)
print(f"\nproject(X均值) 范数={np.linalg.norm(p):.6f}")
print(f"distance={a.distance(x):.6f}")
print(f"potential={a.potential(x):.6f}")

# 再看一个冷却是样本内的点
x2 = X[0]
print(f"\nX[0]: distance={a.distance(x2):.6f} potential={a.potential(x2):.6f}")

# 检查 basis @ basis.T 是否接近 I
print(f"\nbasis 正交性: ||B B^T - I|| = {np.linalg.norm(a.basis @ a.basis.T - np.eye(k)):.6f}")

# 检查 cov 是否可逆
try:
    inv = np.linalg.inv(a.cov + np.eye(k)*1e-6)
    print(f"cov 可逆，条件数={np.linalg.cond(a.cov):.2e}")
except Exception as e:
    print(f"cov 不可逆: {e}")
