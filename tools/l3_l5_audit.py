r"""L3→L5 审计：召回路径到底有没有进决策（2026-11-28）。

只读副本。每步记录：
- `stream._last_recall.path`（L3 召回路径）
- 动作、选中目标
量：
- 召回长度/非空率；
- LEAVE 时，选中目标是否在召回路径里（L3 有没有影响选目标）；
- 召回路径与下一步动作/目标的相关性；
- 相邻两步召回路径的 Jaccard（记忆连续性）。

Run: .\.venv\Scripts\python.exe tools\l3_l5_audit.py
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

STEPS = 400
CP = Path("data/state/_audit_l35.pkl")


def path_of(rec) -> list[int]:
    if rec is None:
        return []
    return [int(x) for x in getattr(rec, "path", []) or []]


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main() -> int:
    shutil.copyfile(Path("data/state/l5.pkl"), CP)
    st = Stream()
    assert persist.load_l5(st, name=CP.name)
    rows = []
    prev = set()
    for _ in range(STEPS):
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        path = path_of(st._last_recall)
        cur = set(path)
        rows.append({
            "len": len(path),
            "action": int(st._action),
            "target": st._leave_target,
            "hit_target": (int(st._leave_target) in cur
                           if st._leave_target is not None else None),
            "jaccard": jaccard(prev, cur),
        })
        prev = cur
    try:
        CP.unlink()
    except FileNotFoundError:
        pass

    lens = np.array([r["len"] for r in rows], dtype=float)
    leave = np.array([r["action"] for r in rows], dtype=float)
    hits = [r["hit_target"] for r in rows if r["hit_target"] is not None]
    jac = np.array([r["jaccard"] for r in rows], dtype=float)
    out = {
        "steps": len(rows),
        "recall_len_mean": round(float(lens.mean()), 3),
        "recall_len_max": int(lens.max()),
        "recall_nonempty_share": round(float((lens > 0).mean()), 3),
        "leave_share": round(float(leave.mean()), 3),
        "target_in_recall_share": round(float(np.mean(hits)), 3) if hits else None,
        "recall_vs_leave_corr": round(float(np.corrcoef(lens, leave)[0, 1]), 4)
        if lens.std() > 1e-9 and leave.std() > 1e-9 else 0.0,
        "path_jaccard_mean": round(float(jac.mean()), 3),
    }
    dest = Path("data/exp/l3_l5_audit.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"summary": out, "rows": rows}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("== L3→L5 审计（副本 %d 步）==" % STEPS)
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"  明细: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
