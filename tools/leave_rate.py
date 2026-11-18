r"""tools/leave_rate.py — 她最近真的在走吗？（精确口径）

为什么需要它
------------
2026-11-18 傍晚，人家两次把「她在走」报成「她站着」：
  1. 数 `events.jsonl` 里 `noise_driven=False` 的转移 —— leave 走一步还在同一区域时不产生转移；
  2. 数 `observations` 窗口内的 leave 条数 —— 每 3 分钟才一条采样，看不出频率。

**精确口径只有一个：读存档的 `action_counts` 两次，算差值。**
这个工具就是干这个的。

用法：
    python tools\leave_rate.py              # 读两次、间隔 120 秒
    python tools\leave_rate.py --seconds 300
    python tools\leave_rate.py --once       # 只看累积值（不等待）

只读存档；不写任何东西。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def snap() -> dict:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no archive")
    acts = dict(st.action_counts)
    return {
        "steps": int(st._step_count),
        "leave": int(acts.get(1, 0)),
        "hold": int(acts.get(0, 0)),
        "t": time.time(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    a0 = snap()
    total = a0["leave"] + a0["hold"]
    print(f"累积: steps={a0['steps']} leave={a0['leave']} hold={a0['hold']} "
          f"leave_share={a0['leave'] / max(1, total):.3f}")
    if a.once:
        return 0

    print(f"等 {a.seconds:.0f} 秒再读一次……")
    time.sleep(a.seconds)
    a1 = snap()
    ds = a1["steps"] - a0["steps"]
    dl = a1["leave"] - a0["leave"]
    dh = a1["hold"] - a0["hold"]
    if ds <= 0:
        print("这 {:.0f} 秒里没有推进（她可能停了）".format(a1["t"] - a0["t"]))
        return 1
    print(f"最近 {a1['t'] - a0['t']:.0f} 秒: steps +{ds}, leave +{dl}, hold +{dh}, "
          f"leave/share={dl / ds:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
