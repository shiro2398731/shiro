r"""B24：真代谢（stream.use_metabolism）的副本对照。

A 原规则；B 原规则+真代谢（移动扣能量、待着回、见底强制歇）。
量：能量最低/结束、到达区域、最大位移、结构增长、leave 占比。

Run: .\.venv\Scripts\python.exe tools\exp\probe_metabolism.py
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

FIXED = os.path.join("data", "state", "_delib_fixed.pkl")
STEPS = 400


def run(mode: str):
    st = Stream()
    assert persist.load_l5(st, name=os.path.basename(FIXED))
    st._home_x = None
    st._leave_target = None
    if mode == "metabolism":
        st.use_metabolism = True
    if mode == "delib_metabolism":
        st.use_metabolism = True
        st.use_deliberation = True
        st.deliberation.value_mode = True
        st.deliberation.energy_mode = True
    x0 = np.asarray(st._st.x, dtype=float).copy()
    r0 = len(st.landscape.state_space.attractors())
    seen = set()
    leaves = 0
    far = 0.0
    emin = 1.0
    for _ in range(STEPS):
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        emin = min(emin, float(st.energy))
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
        "energy_min": round(float(emin), 3),
        "energy_end": round(float(st.energy), 3),
    }


if not os.path.exists(FIXED):
    shutil.copyfile(os.path.join("data", "state", "l5.pkl"), FIXED)
print("rules     ", run("rules"))
print("metabolism", run("metabolism"))
print("delib+met ", run("delib_metabolism"))
