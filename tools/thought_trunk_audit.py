r"""主干重定位后的审计：驱动 pair 场 → 点火（想的内容）。

只读副本 500 步。每步：
- 记录「张力最大的 k 个 pair」的端点集合；
- 记录这一步 workspace 是否点火、点火区域 uid；
- 比较：点火区域落在 top-k pair 上的比例 vs 随机 chance（k*2/N）。

Run: .\.venv\Scripts\python.exe tools/thought_trunk_audit.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 500
CP = Path("data/state/_audit_thought.pkl")


def main() -> int:
    shutil.copyfile(Path("data/state/l5.pkl"), CP)
    st = Stream()
    assert persist.load_l5(st, name=CP.name)
    rows = []
    prev_ign = int(st.workspace.total_ignitions)
    for _ in range(STEPS):
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        ign = int(st.workspace.total_ignitions) - prev_ign
        prev_ign = int(st.workspace.total_ignitions)
        ts = sorted(st.drive.last, key=lambda t: t.amount, reverse=True)
        top = ts[:10]
        pair_sets = {}
        for k in (1, 5, 10):
            s = set()
            for t in top[:k]:
                s.add(int(t.a))
                s.add(int(t.b))
            pair_sets[k] = s
        uid = st.workspace.history[-1].attractor_id if st.workspace.history else None
        rows.append({"ign": int(ign > 0),
                     "uid": None if uid is None else int(uid),
                     **{f"in_top{k}": (uid is not None and int(uid) in pair_sets[k])
                        for k in (1, 5, 10)}})
    try:
        CP.unlink()
    except FileNotFoundError:
        pass
    n_attr = len(st.landscape.state_space.attractors())
    ign_rows = [r for r in rows if r["ign"] and r["uid"] is not None]
    out = {"steps": STEPS, "ignitions": len(ign_rows), "n_attr": n_attr}
    for k in (1, 5, 10):
        hit = sum(1 for r in ign_rows if r[f"in_top{k}"])
        out[f"ign_in_top{k}"] = round(hit / max(len(ign_rows), 1), 3)
        out[f"chance_top{k}"] = round(min(1.0, 2 * k / max(n_attr, 1)), 4)
    dest = Path("data/exp/thought_trunk_audit.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"summary": out, "rows": rows}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("== 想的主干审计（副本 %d 步）==" % STEPS)
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"  明细: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
