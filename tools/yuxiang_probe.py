r"""余响原型：内部漂移写进关系记忆（副本，不碰真机）。

shiromo 的「余响」= 无外部输入时的内部漂移，不记账、不解释、不被重启清除。
这里只试一件事：让内部漂移**碰到关系记忆**（graph），而不是只推位置。

对照（同一存档、同 RNG、500 步）：
- baseline：原样
- drift：每步以 p=0.01 随机 touch 一个已有区域（内部漂移，非外部输入）

量：张力最大 pair 的变化次数/不同 pair 数、关系门（top pair 变化且没说过）会放行几句、
点火、区域数。

Run: .\.venv\Scripts\python.exe tools/yuxiang_probe.py
"""

from __future__ import annotations

import os
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = int(os.environ.get("SHIRO_STEPS", "500"))
P = 0.01
FIXED = Path("data/state/_audit_yuxiang_fixed.pkl")


def run(p: float):
    drift = p > 0
    cp = Path("data/state/_audit_yuxiang.pkl")
    shutil.copyfile(FIXED, cp)
    st = Stream()
    assert persist.load_l5(st, name=cp.name)
    rng = np.random.default_rng(7)
    pairs = []
    pair_by_step = []
    ign_by_step = []
    prev_pair = None
    said = deque(maxlen=64)
    rel_speak = 0
    ign0 = int(st.workspace.total_ignitions)
    r0 = len(st.landscape.state_space.attractors())
    for _ in range(STEPS):
        if drift and rng.random() < p:
            attrs = st.landscape.state_space.attractors()
            if attrs:
                a = attrs[int(rng.integers(0, len(attrs)))]
                st.graph.touch(int(a.uid))
        st.landscape.set_noise_amplitude(0.1957)
        prev_ign = int(st.workspace.total_ignitions)
        st.step()
        ign_by_step.append(int(st.workspace.total_ignitions) - prev_ign)
        ts = sorted(st.drive.last, key=lambda t: t.amount, reverse=True)
        if ts:
            pair = (int(ts[0].a), int(ts[0].b))
            pairs.append(pair)
            if pair != prev_pair:
                if pair not in said:
                    said.append(pair)
                    rel_speak += 1
            prev_pair = pair
        pair_by_step.append(pair if ts else None)
    try:
        cp.unlink()
    except FileNotFoundError:
        pass
    return {
        "regions": len(st.landscape.state_space.attractors()),
        "regions_delta": len(st.landscape.state_space.attractors()) - r0,
        "ignitions": int(st.workspace.total_ignitions) - ign0,
        "distinct_top_pairs": len(set(pairs)),
        "top_pair_changes": sum(1 for a, b in zip(pairs, pairs[1:]) if a != b),
        "relation_would_speak": rel_speak,
        "ign_bins": [sum(ign_by_step[i:i+1000]) for i in range(0, STEPS, 1000)],
        "pair_bins": [len({x for x in pair_by_step[i:i+1000] if x is not None})
                      for i in range(0, STEPS, 1000)],
    }


def main() -> int:
    if not FIXED.exists():
        shutil.copyfile(Path("data/state/l5.pkl"), FIXED)
    p_env = os.environ.get("SHIRO_YUXIANG_P")
    ps = (float(p_env),) if p_env else (0.0, 0.002, 0.005, 0.01, 0.02)
    for p in ps:
        print(f"p={p:<6}", run(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())

