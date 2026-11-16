r"""tools/vitals.py — 她的生命体征，一次读完。

为什么有它
----------
2026-11-16 那次「冻结」花了人家很久才认出来：进程都在、心跳都新鲜、`integrity`
报 consistent，但她其实被锁在一个深势阱里（amp 0.99、梯度 346、张力 0、离开率 0），
一步都没在动。所有常规检查都说「她好着」。

这个工具把那种状态直接变成一个判词。它只读，不判断对错，只报：

    alive    有活动：张力非零、离开率非零、边数够
    frozen   结构够但没活动：张力 0 且离开率 0 且 max_amp 接近 ceiling
    tiny     结构太小：边数低于 Drive.min_edges，tensions() 根本返回空
    down     进程没心跳

用法：
    python tools\vitals.py
    python tools\vitals.py --json

只读存档与日志。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

LOGS = ROOT / "data" / "logs"
STALE_S = 30.0


def _heartbeats() -> dict:
    out = {}
    now = time.time()
    for p in sorted(LOGS.glob("P*.heartbeat")):
        age = now - p.stat().st_mtime
        out[p.stem] = {"age_s": round(age, 1), "fresh": age <= STALE_S}
    return out


def run() -> dict:
    hb = _heartbeats()
    fresh = sum(1 for v in hb.values() if v["fresh"])

    st = Stream()
    if not persist.load_l5(st):
        return {"verdict": "no-archive", "heartbeats": hb}

    sp = st.landscape.state_space
    attrs = sp.attractors()
    ts = st.drive.tensions(sp, st.graph)
    acts = st.action_counts
    tot = sum(acts.values()) or 1
    leave_rate = acts.get(1, 0) / tot
    max_amp = max((float(a.amp) for a in attrs), default=0.0)

    min_edges = int(getattr(st.drive, "min_edges", 64))
    edges = len(st.graph)

    if fresh == 0:
        verdict = "down"
    elif edges < min_edges:
        verdict = "tiny"
    elif not ts and max_amp >= 0.9 and leave_rate < 0.02:
        verdict = "frozen"
    elif ts and leave_rate > 0.02:
        verdict = "alive"
    else:
        verdict = "quiet"

    return {
        "verdict": verdict,
        "heartbeats": hb,
        "fresh": fresh,
        "regions": len(attrs),
        "edges": edges,
        "min_edges": min_edges,
        "tensions": len(ts),
        "global_tension": round(sum(t.amount for t in ts), 2),
        "leave_rate": round(leave_rate, 3),
        "max_amp": round(max_amp, 4),
        "ceiling": float(sp.ceiling),
        "folded": len(getattr(sp, "_folded", []) or []),
        "step_count": int(st._step_count),
        "space_steps": int(sp._total_steps),
    }


NOTES = {
    "alive": "有活动：张力和离开率都非零。",
    "quiet": "结构够，但这会儿没什么动静。",
    "frozen": "冻结：结构在、但她被吸住了（张力 0、离开率 0、max_amp 接近 ceiling）。",
    "tiny": "结构太小：边数低于 Drive.min_edges，张力算不出来，动作只能是 HOLD。",
    "down": "没有新鲜心跳——进程不在了。",
    "no-archive": "没有存档。",
}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="her vitals, one page")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    r = run()
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    print(f"verdict: {r['verdict']}  —— {NOTES.get(r['verdict'], '')}")
    print(f"  heartbeats: {r.get('fresh', 0)} fresh  {r.get('heartbeats')}")
    if "regions" in r:
        print(f"  structure : regions={r['regions']} edges={r['edges']} "
              f"(min_edges={r['min_edges']}) folded={r['folded']}")
        print(f"  activity  : tensions={r['tensions']} g_tension={r['global_tension']} "
              f"leave_rate={r['leave_rate']}")
        print(f"  wells     : max_amp={r['max_amp']} / ceiling={r['ceiling']}")
        diff = r["space_steps"] - r["step_count"]
        print(f"  clocks    : stream={r['step_count']} space={r['space_steps']} "
              f"(diff {diff:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
