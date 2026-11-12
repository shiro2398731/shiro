"""
tools/intervene.py — 她的一步动作到底改变了什么。

两件事，分开测
--------------
1. **她做得到吗**：同一条存档、同一条噪声序列，一边强制留在原处，一边强制离开，
   比较下一步的区域分布。差出来的部分只能是这一步动作（干预，不是相关）。
2. **做了有什么用**：她自己 unresolved tension 在动作之后往哪走。如果"离开"之后张力
   并不比她"留下"时低，那这只手目前只是让她动，还没为她做任何事——那么学策略就没有
   东西可学，先把这件事说清楚，比加一个学习器更有用。

另外测一个依赖关系（只在不强制的那组里成立）：
    P(离开 | 张力高于她自己中位)  vs  P(离开 | 张力低于中位)
现在她的判据只看停留时长，所以这两者**应当相近**——相近反而说明代码没在偷偷按张力办事。

用法
----
    python tools\\intervene.py
    python tools\\intervene.py --warm 800 --steps 1200
    python tools\\intervene.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.action import HOLD, LEAVE  # noqa: E402
from src.L5_cognition.grounding import heldout_gain  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def _fresh(amplitude: float) -> Stream:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive: run P2 first, or point state_dir at one")
    st.landscape.set_noise_amplitude(amplitude)
    return st


def _tension(st: Stream) -> float:
    return float(st.drive.summary().get("sum", 0.0))


def _trace(st: Stream, steps: int, amplitude: float) -> dict:
    ids: list[int] = []
    tensions: list[float] = []
    leaves: list[bool] = []
    x0 = st._st.x.copy()
    for _ in range(steps):
        st.landscape.set_noise_amplitude(amplitude)
        st.step()
        aid = st.detector.current_attractor
        if aid is not None:
            ids.append(int(aid))
        tensions.append(_tension(st))
        leaves.append(st._action == LEAVE)

    stay = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    n = max(len(ids) - 1, 1)
    third = max(len(tensions) // 3, 1)
    t_start = float(np.mean(tensions[:third]))
    t_end = float(np.mean(tensions[-third:]))
    med = float(np.median(tensions)) if tensions else 0.0
    high = [lv for t, lv in zip(tensions, leaves) if t > med]
    low = [lv for t, lv in zip(tensions, leaves) if t <= med]

    return {
        "steps": steps,
        "visits": len(ids),
        "distinct": len(set(ids)),
        "stay_share": round(stay / n, 4),
        "displacement": round(float(np.linalg.norm(st._st.x - x0)), 4),
        "action_counts": {("hold" if k == HOLD else "leave"): v
                          for k, v in st.action_counts.items()},
        "next_pointer_bits": heldout_gain(list(zip(ids, ids[1:]))).get("gain_bits"),
        "tension_start": round(t_start, 5),
        "tension_end": round(t_end, 5),
        "tension_delta": round(t_end - t_start, 5),
        "leave_when_tension_high": round(sum(high) / len(high), 4) if high else None,
        "leave_when_tension_low": round(sum(low) / len(low), 4) if low else None,
        "region_histogram": dict(Counter(ids).most_common(6)),
    }


def run(warm: int = 400, steps: int = 600, amplitude: float = 0.02) -> dict:
    hold = _fresh(amplitude)
    leave = _fresh(amplitude)
    for _ in range(warm):                      # identical warm-up, no forcing
        for st in (hold, leave):
            st.landscape.set_noise_amplitude(amplitude)
            st.step()

    hold.force_action = HOLD
    leave.force_action = LEAVE
    a = _trace(hold, steps, amplitude)
    b = _trace(leave, steps, amplitude)

    free = _fresh(amplitude)
    for _ in range(warm):
        free.landscape.set_noise_amplitude(amplitude)
        free.step()
    c = _trace(free, steps, amplitude)

    return {
        "warm": warm,
        "steps": steps,
        "amplitude": amplitude,
        "do_hold": a,
        "do_leave": b,
        "unforced": c,
        "consequence": {
            "tension_delta_hold": a["tension_delta"],
            "tension_delta_leave": b["tension_delta"],
            "leave_helps": (b["tension_delta"] < a["tension_delta"]),
        },
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="what her action actually changes")
    ap.add_argument("--warm", type=int, default=400)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--amplitude", type=float, default=0.02)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    r = run(args.warm, args.steps, args.amplitude)
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    print(f"warm={r['warm']}  steps={r['steps']}  amplitude={r['amplitude']}"
          f"  (same checkpoint, same noise sequence)")
    cols = (("do(hold)", r["do_hold"]), ("do(leave)", r["do_leave"]),
            ("unforced", r["unforced"]))
    keys = (("distinct", "distinct regions"),
            ("stay_share", "P(next == current)"),
            ("displacement", "|x - x0|"),
            ("tension_start", "tension (first third)"),
            ("tension_end", "tension (last third)"),
            ("tension_delta", "tension delta"),
            ("leave_when_tension_high", "P(leave | tension > median)"),
            ("leave_when_tension_low", "P(leave | tension <= median)"),
            ("action_counts", "actions taken"))
    print(f"  {'':<30}" + "".join(f"{c[0]:>16}" for c in cols))
    for key, label in keys:
        row = f"  {label:<30}"
        for _, d in cols:
            row += f"{str(d.get(key)):>16}"
        print(row)

    print(f"consequence: tension delta hold={r['consequence']['tension_delta_hold']:+.5f}  "
          f"leave={r['consequence']['tension_delta_leave']:+.5f}  "
          f"leave reduces tension: {r['consequence']['leave_helps']}")
    print("Reading: if leave does not reduce tension relative to hold, her hand moves")
    print("         her but does nothing for her -- there is nothing for a policy to learn.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
