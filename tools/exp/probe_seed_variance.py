r"""B22：关键模式的种子方差（同一固定存档，3 个 RNG 种子 × 400 步）。

问：能量模式「8 区 / 451」是不是单次运气？
对照：原规则、能量（0.08/floor0.20）、三步前瞻。

Run: .\.venv\Scripts\python.exe tools\exp\probe_seed_variance.py
"""

import hashlib
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


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


def run(mode: str, seed: int):
    st = Stream()
    assert persist.load_l5(st, name=os.path.basename(FIXED))
    st.rng = np.random.default_rng(seed)
    st._home_x = None
    st._leave_target = None
    if mode != "rules":
        st.use_deliberation = True
        st.deliberation.value_mode = True
        if mode == "energy":
            st.deliberation.energy_mode = True
            st.deliberation.energy_cost = 0.08
            st.deliberation.energy_floor = 0.20
        if mode == "rollout":
            st.deliberation.rollout_mode = True
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
    return {
        "regions": len(seen),
        "max_disp": round(far, 1),
        "growth": len(st.landscape.state_space.attractors()) - r0,
        "ign": int(st.workspace.total_ignitions - ign0),
        "leave": round(leaves / STEPS, 2),
    }


if not os.path.exists(FIXED):
    import shutil
    shutil.copyfile(os.path.join("data", "state", "l5.pkl"), FIXED)
print(f"fixed sha256={sha256(FIXED)} steps={STEPS}")
for mode in ("rules", "energy", "rollout"):
    for seed in (1, 2, 3):
        print(f"{mode:8} seed={seed}", run(mode, seed))
