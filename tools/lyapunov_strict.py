r"""tools/lyapunov_strict.py — 严格算 Lyapunov 指数（确定性 + 两条轨迹）。

标准定义（动力系统教材 / Strogatz 的「two trajectories」方法）：
    λ = lim_{T->∞} (1/T) ln( |δx(T)| / |δx(0)| )

严格条件：
  1. 确定性 —— 关掉噪声（set_noise_amplitude(0)）。
     之前人家用「不同种子噪声」，那测的是「含噪声敏感性」，不是 Lyapunov。
  2. 两条轨迹从同一个 x 出发，差 δx0（很小）。
  3. 每一步量 |δx(t)|，取 ln 平均。

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
    # 严格：关噪声（确定性）。之前用 0.02 是错的——测的是「含噪声敏感性」。
    st.landscape.set_noise_amplitude(0.0)
    return st


def lyapunov(seed: int, steps: int, delta0: float, samples: np.ndarray, from_zero: bool) -> tuple[float, np.ndarray]:
    a = make(seed, from_zero, samples)
    b = make(seed, from_zero, samples)   # 同一个种子、同一个起点

    x0 = np.asarray(a._st.x, dtype=float).copy()
    # 两个轨迹同起点，其中一个加 δx0（沿一个固定方向）
    rng = np.random.default_rng(seed * 7919 + 11)
    v = rng.normal(0, 1, a.D)
    v = v / np.linalg.norm(v) * delta0
    b._st.x = x0 + v

    logs: list[float] = []
    for _ in range(steps):
        a.step()
        b.step()
        d = float(np.linalg.norm(np.asarray(a._st.x) - np.asarray(b._st.x)))
        if d > 0:
            logs.append(float(np.log(d / delta0)))
    lam = float(np.mean(logs)) if logs else float("nan")
    return lam, np.asarray(logs)


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    delta0 = 1e-7
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, (64, 128))

    print("严格 Lyapunov（确定性 / 关噪声 / 两条轨迹）")
    print(f"steps={steps}  delta0={delta0}")
    print()
    for from_zero, name in [(True, "从零"), (False, "真机存档")]:
        lam, logs = lyapunov(1, steps, delta0, samples, from_zero)
        print(f"[{name}] λ = {lam:+.6f} / step   (n={len(logs)})")
        # 分段看前 1/3 和后 1/3（看是否收敛）
        k = max(1, len(logs) // 3)
        print(f"        前 1/3 平均 {np.mean(logs[:k]):+.4f}   后 1/3 平均 {np.mean(logs[-k:]):+.4f}")
    print()
    print("λ>0 = 混沌 (差异指数放大)   λ≈0 = 中性   λ<0 = 收敛 (差异衰减)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
