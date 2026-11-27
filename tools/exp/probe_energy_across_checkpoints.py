r"""B23：能量模式换 3 个历史存档（同格式备份）的稳健性。

同一个起点配对已经做过；这里问：换不同日期的存档，能量模式还成立吗？
对照原规则。每个存档 400 步。

Run: .\.venv\Scripts\python.exe tools\exp\probe_energy_across_checkpoints.py
"""

import os
import sys

import numpy as np

ROOT = r"C:\neolaunchertest\3\bot-3021443116\shiro_alice"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.action import LEAVE  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 400
CKS = [
    "l5.pre-attnacc-0006.pkl",
    "l5.pre-igntotal-0019.pkl",
    "l5.pre-recent64-1249.pkl",
]


def run(ck: str, mode: str):
    st = Stream()
    assert persist.load_l5(st, name=ck)
    st._home_x = None
    st._leave_target = None
    if mode == "energy":
        st.use_deliberation = True
        st.deliberation.value_mode = True
        st.deliberation.energy_mode = True
        st.deliberation.energy_cost = 0.08
        st.deliberation.energy_floor = 0.20
    x0 = np.asarray(st._st.x, dtype=float).copy()
    r0 = len(st.landscape.state_space.attractors())
    seen = set()
    leaves = 0
    far = 0.0
    for _ in range(STEPS):
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        leaves += int(int(st._action) == LEAVE)
        aid = st.detector.current_attractor
        if aid is not None:
            seen.add(int(aid))
        far = max(far, float(np.linalg.norm(np.asarray(st._st.x, dtype=float) - x0)))
    return {
        "regions": len(seen),
        "max_disp": round(far, 1),
        "growth": len(st.landscape.state_space.attractors()) - r0,
        "leave": round(leaves / STEPS, 2),
    }


for ck in CKS:
    if not os.path.exists(os.path.join("data", "state", ck)):
        print("missing", ck)
        continue
    print(ck)
    print("  rules ", run(ck, "rules"))
    print("  energy", run(ck, "energy"))
