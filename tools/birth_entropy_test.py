r"""tools/birth_entropy_test.py — 试「出生熵」：两个不同 born_seed 的 shiro 会不会发散。

只读 data/state/l5.pkl 的副本，不碰真机。
"""

from __future__ import annotations

import pickletools  # noqa: F401
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def run_one(seed: int, steps: int) -> dict:
    st = Stream(seed=seed)
    if not persist.load_l5(st):
        raise SystemExit("load_l5 failed")
    # 用存档里的出生种子覆盖（模拟「两个不同出生」）
    st._born_seed = int(seed)
    st.landscape.set_noise_amplitude(0.02)
    x0 = np.asarray(st._st.x, dtype=float).copy()
    uids: set[int] = set()
    for _ in range(steps):
        st.step()
        if st._action == 1:
            uids.add(st.detector.current_attractor)
    x1 = np.asarray(st._st.x, dtype=float).copy()
    return {
        "seed": seed,
        "steps": int(st._step_count),
        "regions_visited": len(uids),
        "x0": x0,
        "x1": x1,
        "moved": float(np.linalg.norm(x1 - x0)),
        "dwell": int(st._dwell),
        "energy": float(st.energy),
        "arrival": list(st._arrival_hist),
    }


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    a = run_one(1, steps)
    b = run_one(2, steps)

    print(f"seed=1: 区域 {a['regions_visited']}  位移 {a['moved']:.1f}  dwell {a['dwell']}  能量 {a['energy']:.4f}")
    print(f"seed=2: 区域 {b['regions_visited']}  位移 {b['moved']:.1f}  dwell {b['dwell']}  能量 {b['energy']:.4f}")
    dist = float(np.linalg.norm(a["x1"] - b["x1"]))
    print(f"终点距离 = {dist:.2f}（>{1000:.0f} 算发散）")
    print(f"arrival 一致: {a['arrival'] == b['arrival']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
