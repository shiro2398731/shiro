r"""B 选项最小验证：去掉「一步滞后」会不会改变行为（只读副本）。

现状：`_choose_action` 用的是上一步末尾算出的 `_tension_now`；
本实验在同一步开始时重算张力场，再决定动作，比较动作序列。

Run: .\.venv\Scripts\python.exe tools\same_state_probe.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.action import HOLD, LEAVE  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 500
FIXED = Path("data/state/_audit_same_state_fixed.pkl")


def run(same_step: bool):
    cp = Path("data/state/_audit_same_state.pkl")
    shutil.copyfile(FIXED, cp)
    st = Stream()
    assert persist.load_l5(st, name=cp.name)
    if same_step:
        def choose():
            # 同一步：先按当前图重算张力，再用它决定
            st.drive.tensions(st.landscape.state_space, st.graph)
            st._tension_now = float(st.drive.summary().get("sum", 0.0))
            action = st.actions.choose(st._tension_now, list(st._tension_history))
            st._tension_history.append(st._tension_now)
            st.action_counts[action] = st.action_counts.get(action, 0) + 1
            if action == LEAVE:
                st._apply_leave()
            return action
        st._choose_action = choose
    actions = []
    sums = []
    deltas = []
    for _ in range(STEPS):
        prev = float(st._tension_now)
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        actions.append(int(st._action))
        sums.append(float(st._tension_now))
        deltas.append(abs(float(st._tension_now) - prev))
    try:
        cp.unlink()
    except FileNotFoundError:
        pass
    return {
        "actions": actions,
        "leave_share": round(float(np.mean(actions)), 3),
        "tension_mean": round(float(np.mean(sums)), 2),
        "delta_abs_mean": round(float(np.mean(deltas)), 3),
    }


def main() -> int:
    if not FIXED.exists():
        shutil.copyfile(Path("data/state/l5.pkl"), FIXED)
    a = run(False)
    b = run(True)
    hamming = sum(1 for x, y in zip(a["actions"], b["actions"]) if x != y)
    print("baseline ", {k: v for k, v in a.items() if k != "actions"})
    print("same_step", {k: v for k, v in b.items() if k != "actions"})
    print("action_hamming", hamming, "share", round(hamming / STEPS, 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
