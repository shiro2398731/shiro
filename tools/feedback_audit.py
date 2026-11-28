r"""反馈审计：动作/结果到底有没有改变下一步的驱动与图（只读副本）。

B 选项判据 2 的基线测量：
- action_t 与 Δtension_{t+1} 的相关；
- 位移_t 与 Δedges_{t+1} 的相关；
- 与「打乱动作顺序」的零分布比较（看是否只是共同漂移）。

Run: .\.venv\Scripts\python.exe tools\feedback_audit.py
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

STEPS = 500
CP = Path("data/state/_audit_feedback.pkl")


def corr(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    shutil.copyfile(Path("data/state/l5.pkl"), CP)
    st = Stream()
    assert persist.load_l5(st, name=CP.name)
    rows = []
    prev_sum = float(st._tension_now)
    for _ in range(STEPS):
        x0 = np.asarray(st._st.x, dtype=float).copy()
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        cur_sum = float(st._tension_now)
        disp = float(np.linalg.norm(np.asarray(st._st.x, dtype=float) - x0))
        rows.append({
            "action": int(st._action),
            "disp": disp,
            "dsum": cur_sum - prev_sum,
            "edges": len(st.graph),
            "regions": len(st.landscape.state_space.attractors()),
        })
        prev_sum = cur_sum
    try:
        CP.unlink()
    except FileNotFoundError:
        pass

    act = np.array([r["action"] for r in rows], dtype=float)
    disp = np.array([r["disp"] for r in rows], dtype=float)
    dsum = np.array([r["dsum"] for r in rows], dtype=float)
    edges = np.array([r["edges"] for r in rows], dtype=float)
    dedges = np.diff(edges, prepend=edges[0])
    # 动作/位移 t 与下一步变化 t+1
    a2d = corr(act[:-1], dsum[1:])
    d2e = corr(disp[:-1], dedges[1:])
    # 零分布：打乱动作顺序，重算相关
    rng = np.random.default_rng(0)
    null = []
    for _ in range(200):
        perm = rng.permutation(act[:-1])
        null.append(corr(perm, dsum[1:]))
    null = np.asarray(null)
    out = {
        "steps": STEPS,
        "leave_share": round(float(act.mean()), 3),
        "action_to_next_dsum": round(a2d, 4),
        "disp_to_next_dedges": round(d2e, 4),
        "null_action_dsum_mean": round(float(null.mean()), 4),
        "null_action_dsum_p95": round(float(np.percentile(null, 95)), 4),
        "action_effect_z": round(float((a2d - null.mean()) / max(null.std(), 1e-9)), 3),
    }
    dest = Path("data/exp/feedback_audit.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"summary": out, "rows": rows}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("== 反馈审计（副本 %d 步）==" % STEPS)
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"  明细: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
