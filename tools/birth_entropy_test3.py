r"""tools/birth_entropy_test3.py — 给 x 加「从 seed 来的小扰动」，看会不会「混沌放大」。

不碰真机。只测：
  - 同一份校准（同一个「环境」）
  - 不同 seed -> x 加 ±1e-3 的扰动（出生点不同）
  - 跑 N 步 -> 看两个的 x 距离
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.stream import Stream  # noqa: E402


def run_with_perturb(seed: int, steps: int, samples: np.ndarray, eps: float) -> dict:
    st = Stream(dim=128, gamma=0.05, dt=0.01, seed=seed)
    st.calibrate(samples)
    st.landscape.set_noise_amplitude(0.02)
    # 出生扰动：从 seed 的 rng 来，±eps
    born_rng = np.random.default_rng(seed * 1000003 + 7)
    x0 = np.asarray(st._st.x, dtype=float).copy()
    x0 = x0 + born_rng.normal(0, eps, st.D)
    st._st.x = x0.copy()
    for _ in range(steps):
        st.step()
    x1 = np.asarray(st._st.x, dtype=float).copy()
    return {"seed": seed, "x0": x0, "x1": x1, "moved": float(np.linalg.norm(x1 - x0))}


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, (64, 128))

    a = run_with_perturb(1, steps, samples, eps)
    b = run_with_perturb(2, steps, samples, eps)

    d0 = float(np.linalg.norm(a["x0"] - b["x0"]))
    d1 = float(np.linalg.norm(a["x1"] - b["x1"]))
    ratio = d1 / d0 if d0 > 0 else float("inf")
    print(f"eps={eps}  steps={steps}")
    print(f"起点距离 = {d0:.6f}")
    print(f"终点距离 = {d1:.2f}")
    print(f"放大倍数 = {ratio:.1f}x")
    print(f"位移 seed=1: {a['moved']:.1f}   seed=2: {b['moved']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
