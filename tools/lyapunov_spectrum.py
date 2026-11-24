r"""tools/lyapunov_spectrum.py — Lyapunov 谱（128 个），用标准 Benettin 算法。

标准来源：
  Benettin, Galgani, Giorgilli, Strelcyn (1980), "Lyapunov characteristic exponents
  for smooth dynamical systems and for Hamiltonian systems; a method for computing
  all of them. Part 1: Theory"

方法：
  1. 一个基准轨迹 + n 个正交的扰动向量（初始 δx0 e_i）
  2. 每步：所有扰动向量按动力学演化，量长度，取 ln 累加
  3. 定期 Gram-Schmidt 正交化（防止所有向量都朝最大方向）
  4. λ_i = (1/T) Σ ln(|δx_i(t)| / δx0)

局限：
  - 这里用「两条轨迹差」近似「切向量」的演化（不是解析 Jacobian）——
    对小 δx0 成立（线性区）
  - 128 维、2000 步、float64

不碰真机（只读副本）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.stream import Stream  # noqa: E402


def make(seed: int, from_zero: bool, samples: np.ndarray) -> Stream:
    st = Stream(dim=128, gamma=0.05, dt=0.01, seed=seed)
    if from_zero:
        st.calibrate(samples)
    else:
        from src.common import persist  # noqa: PLC0415
        persist.load_l5(st)
    st.landscape.set_noise_amplitude(0.0)   # 确定性
    return st


def spectrum(seed: int, steps: int, delta0: float, samples: np.ndarray,
             from_zero: bool, qr_every: int = 10) -> np.ndarray:
    base = make(seed, from_zero, samples)
    dim = base.D

    # 128 个正交扰动向量（单位长度 * delta0）
    Q = np.eye(dim) * delta0
    sums = np.zeros(dim)
    count = 0

    # 用「副本」来量每个扰动向量的演化：先把基准 x 存下
    for t in range(steps):
        x_ref = np.asarray(base._st.x, dtype=float).copy()
        base.step()

        # 对每个扰动向量：单独一条轨迹从 x_ref + Q[:, i] 出发，走一步
        # （128 次 step 太贵——所以按线性近似：Δx = J·δ，用「一份轨迹 + 差分」近似）
        #
        # 简化（仍是标准做法的近似）：用「基准轨迹的一步差」q_i 的「方向保持」：
        # 这里改用「基准轨迹 x 的位移」当「切向量的演化」——**这不是严格 J·δ**。
        # 严格要 Jacobian 或 128 条副本轨迹。
        #
        # 诚实：这个版本是「近似」，人家标出来。
        dx = np.asarray(base._st.x, dtype=float) - x_ref
        nm = float(np.linalg.norm(dx))
        if nm == 0:
            continue
        # 把「一步位移」当作「切向量的平均方向」——每个 q_i 沿 dx 拉长到相同比例
        Q = Q + np.outer(dx, np.ones(dim)) * 0.0  # 占位（不实际用）
        # 真做法见下：没有 Jacobian 就只能近似。这里直接破掉。
        break

    # 上面发现「没 Jacobian 做不了严格谱」——直接报错，不假装。
    raise RuntimeError(
        "Lyapunov 谱需要 Jacobian 或 128 条副本轨迹。"
        "本文件只搭了骨架，没实现——不假装有结果。"
    )


def main() -> int:
    print("Lyapunov 谱（128 个）需要 Jacobian 或 128 条副本轨迹。")
    print("人家只算了「最大 Lyapunov」（tools/lyapunov_strict.py），谱没算。")
    print("不假装有结果。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
