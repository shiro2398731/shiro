r"""GWT→动作审计（2026-11-28）。

问题：全局工作空间点火（ignition）和「走」到底是不是同一个东西？
只读副本：量点火与动作/下一步动作的列联关系，并按张力边距分层。

Run: .\.venv\Scripts\python.exe tools\gwt_action_audit.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 300
CP = Path("data/state/_audit_gwt.pkl")


def rate(xs) -> float:
    return round(float(np.mean(xs)), 3) if len(xs) else 0.0


def main() -> int:
    shutil.copyfile(Path("data/state/l5.pkl"), CP)
    st = Stream()
    assert persist.load_l5(st, name=CP.name)
    rows = []
    prev_ign = st.workspace.total_ignitions
    for _ in range(STEPS):
        past = list(st._tension_history)
        med = float(sorted(past)[len(past) // 2]) if past else float("nan")
        tension = float(st._tension_now)
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        ign = int(st.workspace.total_ignitions) - prev_ign
        prev_ign = int(st.workspace.total_ignitions)
        rows.append({
            "ign": int(ign > 0),
            "action": int(st._action),
            "margin_pos": int(np.isfinite(med) and tension > med),
        })
    try:
        CP.unlink()
    except FileNotFoundError:
        pass
    nxt = [r["action"] for r in rows[1:]]
    # 当前步点火 vs 当前/下一步动作
    cur_ign = [r for r in rows]
    out = {
        "steps": len(rows),
        "ignitions": int(sum(r["ign"] for r in rows)),
        "leave_share": rate([r["action"] for r in rows]),
        "p_leave_given_ign": rate([r["action"] for r in cur_ign if r["ign"]]),
        "p_leave_given_no_ign": rate([r["action"] for r in cur_ign if not r["ign"]]),
        "p_ign_given_margin_pos": rate([r["ign"] for r in rows if r["margin_pos"]]),
        "p_ign_given_margin_neg": rate([r["ign"] for r in rows if not r["margin_pos"]]),
        "leave_next_after_ign": rate([nxt[i] for i in range(len(nxt)) if rows[i]["ign"]]),
        "leave_next_after_no_ign": rate([nxt[i] for i in range(len(nxt)) if not rows[i]["ign"]]),
    }
    # 同一张力边距层内比较：margin 为正时，点火/不点火后的下一步离开率
    pos = [(rows[i]["ign"], nxt[i]) for i in range(len(nxt)) if rows[i]["margin_pos"]]
    out["pos_margin_leave_next_ign"] = rate([b for a, b in pos if a])
    out["pos_margin_leave_next_no_ign"] = rate([b for a, b in pos if not a])
    dest = Path("data/exp/gwt_action_audit.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"summary": out, "rows": rows}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("== GWT→动作审计（副本 %d 步）==" % STEPS)
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"  明细: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
