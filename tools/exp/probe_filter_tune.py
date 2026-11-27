r"""B16：过滤半径调参（副本 1200 步，只读真机）。

候选过滤已证明能扩区（98 区、结构 +90），但目标 churn 88-92 太高。
这里比较不同过滤半径（以 σ 计）能否压 churn：2σ/3σ/4σ/6σ。

Run: .\.venv\Scripts\python.exe data\exp\probe_filter_tune.py
"""

import os
import shutil
import sys
from collections import Counter

import numpy as np

ROOT = r"C:\neolaunchertest\3\bot-3021443116\shiro_alice"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 1200


def run(sigmas: float):
    cp = "_audit_tune.pkl"
    shutil.copyfile(os.path.join("data", "state", "l5.pkl"),
                    os.path.join("data", "state", cp))
    try:
        st = Stream()
        assert persist.load_l5(st, name=cp)
        st._home_x = None
        st._leave_target = None
        st.filter_targets_by_home = bool(sigmas)
        st.target_filter_sigmas = float(sigmas) if sigmas else 4.0
        seen = set()
        targets = set()
        switches = 0
        prev = None
        far = 0.0
        ign0 = st.workspace.total_ignitions
        r0 = len(st.landscape.state_space.attractors())
        for _ in range(STEPS):
            st.landscape.set_noise_amplitude(0.1957)
            st.step()
            aid = st.detector.current_attractor
            if aid is not None:
                seen.add(int(aid))
            t = st._leave_target
            if t is not None:
                targets.add(int(t))
            if prev is not None and t is not None and t != prev:
                switches += 1
            prev = t
            if st._home_x is not None:
                far = max(far, float(np.linalg.norm(st._home_x - st._st.x)))
        return {
            "regions_seen": len(seen),
            "targets": len(targets),
            "target_switches": switches,
            "max_home": round(far, 1),
            "new_ignitions": int(st.workspace.total_ignitions - ign0),
            "growth": len(st.landscape.state_space.attractors()) - r0,
        }
    finally:
        p = os.path.join("data", "state", cp)
        if os.path.exists(p):
            os.remove(p)


print("baseline ", run(0.0))
for s in (2.0, 3.0, 4.0, 6.0):
    print(f"filter {s}σ", run(s))
