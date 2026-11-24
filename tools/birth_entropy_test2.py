r"""tools/birth_entropy_test2.py — 从零出生（不加载存档）看两个 seed 发散。

对比：
  A. 加载同一个存档 + 改 seed（半路改种子）
  B. 从零 + 同一份校准样本 + 不同 seed（真出生）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.stream import Stream  # noqa: E402


def run_from_zero(seed: int, steps: int, samples: np.ndarray) -> dict:
    st = Stream(dim=128, gamma=0.05, dt=0.01, seed=seed)
    st.calibrate(samples)                      # 同一份校准 = 同一份「环境」
    st.landscape.set_noise_amplitude(0.02)
    x0 = np.asarray(st._st.x, dtype=float).copy()
    uids: set[int] = set()
    for _ in range(steps):
        st.step()
        if st._action == 1:
            uids.add(st.detector.current_attractor)
    x1 = np.asarray(st._st.x, dtype=float).copy()
    return {
        "regions": len(uids),
        "regions_total": len(st.landscape.state_space.attractors()),
        "x0": x0,
        "x1": x1,
        "moved": float(np.linalg.norm(x1 - x0)),
        "energy": float(st.energy),
        "dwell": int(st._dwell),
    }


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rng = np.random.default_rng(0)
    samples = rng.normal(0, 1, (64, 128))       # 同一份「环境」

    a = run_from_zero(1, steps, samples)
    b = run_from_zero(2, steps, samples)

    print(f"seed=1: 区域 {a['regions']}/{a['regions_total']}  位移 {a['moved']:.1f}  dwell {a['dwell']}  能量 {a['energy']:.4f}")
    print(f"seed=2: 区域 {b['regions']}/{b['regions_total']}  位移 {b['moved']:.1f}  dwell {b['dwell']}  能量 {b['energy']:.4f}")
    print(f"起点距离 = {float(np.linalg.norm(a['x0'] - b['x0'])):.2f}（出生点不同？）")
    print(f"终点距离 = {float(np.linalg.norm(a['x1'] - b['x1'])):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
