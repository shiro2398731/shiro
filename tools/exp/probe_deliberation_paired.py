r"""B21：固定存档 + 同一 RNG 的配对比较（副本 400 步）。

每个模式都从同一份 `_delib_fixed.pkl` 起跑，噪声序列一致，
差异只来自「用哪套决定」。

Run: .\.venv\Scripts\python.exe data\exp\probe_deliberation_paired.py
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

FIXED = os.path.join("data", "state", "_delib_fixed.pkl")
STEPS = 400


def run(mode: str):
    st = Stream()
    assert persist.load_l5(st, name=os.path.basename(FIXED))
    st._home_x = None
    st._leave_target = None
    if mode != "rules":
        st.use_deliberation = True
        st.deliberation.value_mode = mode in (
            "value", "energy", "energy_lo", "energy_hi", "energy_deep", "energy_mid")
        st.deliberation.energy_mode = mode in (
            "energy", "energy_lo", "energy_hi", "energy_deep", "energy_mid")
        if mode == "energy_lo":
            st.deliberation.energy_cost = 0.05
        if mode == "energy_hi":
            st.deliberation.energy_cost = 0.15
        if mode == "energy_deep":
            st.deliberation.energy_cost = 0.03
            st.deliberation.energy_floor = 0.15
            st.deliberation.energy_arrive = 0.02
        if mode == "energy_mid":
            st.deliberation.energy_cost = 0.08
            st.deliberation.energy_floor = 0.20
    x0 = np.asarray(st._st.x, dtype=float).copy()
    r0 = len(st.landscape.state_space.attractors())
    ign0 = st.workspace.total_ignitions
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
    out = {
        "regions_seen": len(seen),
        "max_disp": round(far, 1),
        "growth": len(st.landscape.state_space.attractors()) - r0,
        "new_ignitions": int(st.workspace.total_ignitions - ign0),
        "leave_share": round(leaves / STEPS, 3),
    }
    if mode != "rules":
        d = st.deliberation
        out["energy"] = round(d.energy, 3)
        out["intent"] = None if d.intent is None else d.intent["uid"]
    return out


for m in ("rules", "delib", "value", "energy", "energy_lo", "energy_hi",
          "energy_deep", "energy_mid"):
    print(f"{m:11}", run(m))
