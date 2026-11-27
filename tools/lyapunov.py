r"""tools/lyapunov.py — 算她的 Lyapunov 指数（真·严格测）。

原理：
  λ = lim (1/T) Σ log( |δx(t)| / |δx(0)| )
  两条轨迹从同一个 x 出发，差 δx0；每步算两轨迹的距离，取 log 平均。
  λ > 0 = 混沌（差异指数放大）；λ ≈ 0 = 中性；λ < 0 = 收敛（差异衰减）。

不碰真机（只读）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.stream import Stream  # noqa: E402


def lyapunov(seed: int, steps: int, delta0: float, samples: np.ndarray, from_zero: bool) -> tuple[float, list[float]]:
    def make(perturb: bool) -> Stream:
        st = Stream(dim=128, gamma=0.05, dt=0.01, seed=seed)
        if from_zero:
            st.calibrate(samples)
        else:
            from src.common import persist  # noqa: PLC0415
            persist.load_l5(st)
        st.landscape.set_noise_amplitude(0.02)
        if perturb:
            rng = np.random.default_rng(seed * 99991 + 3)
            v = rng.normal(0, 1, st.D)
            v = v / np.linalg.norm(v) * delta0
            x = np.asarray(st._st.x, dtype=float)
            st._st.x = x + v
        return st

    a = make(False)
    b = make(True)

    # 关掉噪声？—— 不关。真实系统有噪声；噪声本身会让两条轨迹分开。
    # 严格的 Lyapunov 应该「同一噪声序列」才公平。这里用「不同种子」的噪声，
    # 所以测的是「含噪声的敏感性」——读的时候要记得这一点。
    logs: list[float] = []
    for _ in range(steps):
        a.step()
        b.step()
        d = float(np.linalg.norm(np.asarray(a._st.x) - np.asarray(b._st.x)))
        if d > 0:
            logs.append(float(np.log(d / delta0)))
    lam = (sum(logs) / len(logs)) if logs else float("nan")
    return lam, logs


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    delta0 = 1e-6
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, (64, 128))

    print(f"steps={steps} delta0={delta0}")
    for from_zero, name in [(True, "从零"), (False, "真机存档")]:
        lam, logs = lyapunov(1, steps, delta0, samples, from_zero)
        print(f"[{name}] λ = {lam:+.4f} / step   样本数 {len(logs)}")
    print()
    print("λ>0 → 混沌（差异指数放大）；λ≈0 → 中性；λ<0 → 收敛（差异衰减）")
    print("注意：两条轨迹用了不同种子噪声，测的是「含噪声敏感性」，不是严格「同一噪声下的 Lyapunov」。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
