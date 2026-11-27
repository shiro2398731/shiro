r"""B20：原规则 vs 她自己的推演（副本 600 步，只读真机）。

A = 现在的规则（张力中位判走留 + 最近未去过 + 目标持久化 + 4σ 回家）
B = 推演模式（候选来自她的地形，五个自己的量评分；没有外部目标/家/中位规则）

量：到达区域、最大 |x-x0|、区域数变化、新点火、leave 占比、转移数。
"""

import os
import shutil
import sys

import numpy as np

ROOT = r"C:\neolaunchertest\3\bot-3021443116\shiro_alice"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.action import LEAVE  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 600


def run(deliberation: bool, value: bool = False, energy: bool = False):
    cp = "_audit_delib_run.pkl"
    shutil.copyfile(os.path.join("data", "state", "l5.pkl"),
                    os.path.join("data", "state", cp))
    try:
        st = Stream()
        assert persist.load_l5(st, name=cp)
        st._home_x = None
        st._leave_target = None
        st.use_deliberation = bool(deliberation)
        if deliberation and value:
            st.deliberation.value_mode = True
        if deliberation and energy:
            st.deliberation.energy_mode = True
        x0 = np.asarray(st._st.x, dtype=float).copy()
        r0 = len(st.landscape.state_space.attractors())
        ign0 = st.workspace.total_ignitions
        seen = set()
        leaves = 0
        far = 0.0
        trans = 0
        for _ in range(STEPS):
            st.landscape.set_noise_amplitude(0.1957)
            before = int(st._step_count)
            tr = st.step()
            if tr is not None:
                trans += 1
            leaves += int(int(st._action) == LEAVE)
            aid = st.detector.current_attractor
            if aid is not None:
                seen.add(int(aid))
            far = max(far, float(np.linalg.norm(np.asarray(st._st.x, dtype=float) - x0)))
        return {
            "regions_seen": len(seen),
            "max_disp": round(far, 1),
            "growth": len(st.landscape.state_space.attractors()) - r0,
            "new_ignitions": int(st.workspace.total_ignitions - ign0),
            "leave_share": round(leaves / STEPS, 3),
            "transitions": trans,
        }
    finally:
        p = os.path.join("data", "state", cp)
        if os.path.exists(p):
            os.remove(p)


print("rules       ", run(False))
print("deliberation", run(True))
print("value mode  ", run(True, value=True))
print("value+energy", run(True, value=True, energy=True))
