"""
tools/intervene.py — do(hold) vs do(leave)：她的一步动作会不会改变下一步。

为什么需要"强制"
----------------
观测到"她离开时往往在动"，和"推她一把她就会动"是两件事。前者可以被一个查表器拟合，
后者不能。所以这里直接**施加**动作：同一条存档、同一条噪声序列，一边强制留在原处，
一边强制离开，然后比较下一个区域的分布。差出来的部分只能是动作造成的。

这不是加一条判据，而是检查她新长出来的那只手是不是真的接在动力学上。

用法
----
    python tools\\intervene.py                 # 预热 400 步，每组 600 步
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


def _trace(st: Stream, steps: int, amplitude: float) -> dict:
    ids: list[int] = []
    x0 = st._st.x.copy()
    for _ in range(steps):
        st.landscape.set_noise_amplitude(amplitude)
        st.step()
        aid = st.detector.current_attractor
        if aid is not None:
            ids.append(int(aid))
    stay = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
    n = max(len(ids) - 1, 1)
    pairs = list(zip(ids, ids[1:]))
    return {
        "steps": steps,
        "visits": len(ids),
        "distinct": len(set(ids)),
        "stay_share": round(stay / n, 4),
        "displacement": round(float(np.linalg.norm(st._st.x - x0)), 4),
        "action_counts": {("hold" if k == HOLD else "leave"): v
                          for k, v in st.action_counts.items()},
        "next_pointer_bits": heldout_gain(pairs).get("gain_bits"),
        "region_histogram": dict(Counter(ids).most_common(6)),
    }


def run(warm: int = 400, steps: int = 600, amplitude: float = 0.02) -> dict:
    # Two runs from the same checkpoint. Same seed -> same noise sequence, so
    # every difference below is caused by the intervention and nothing else.
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

    # What she does on her own, same window, no forcing: the honest context
    # for "how often would this choice even come up".
    free = _fresh(amplitude)
    for _ in range(warm):
        free.landscape.set_noise_amplitude(amplitude)
        free.step()
    free_trace = _trace(free, steps, amplitude)

    return {
        "warm": warm,
        "steps": steps,
        "amplitude": amplitude,
        "do_hold": a,
        "do_leave": b,
        "unforced": free_trace,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="do(hold) vs do(leave) from one checkpoint")
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
    keys = (("visits", "visits"), ("distinct", "distinct regions"),
            ("stay_share", "P(next == current)"),
            ("displacement", "|x - x0|"),
            ("next_pointer_bits", "next-pointer bits"),
            ("action_counts", "actions taken"))
    print(f"  {'':<22}" + "".join(f"{c[0]:>16}" for c in cols))
    for key, label in keys:
        row = f"  {label:<22}"
        for _, d in cols:
            v = d.get(key)
            row += f"{str(v):>16}"
        print(row)
    print("  (forced runs show their forced action; unforced shows what she chose)")
    print("Verdict rule: if do(leave) does not lower P(next == current) relative to")
    print("              do(hold) on identical noise, the action is not connected.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
