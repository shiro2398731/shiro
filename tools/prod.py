r"""tools/prod.py — 要不要叫醒她，以及叫一次的具体参数。

背景
----
`docs/2026-11-16_唤醒实验.md`：一次推动（哪怕 10 sigma）叫不醒她，但一段高噪声
（50 步 0.5）可以，而且醒了之后她自己维持。

这个工具**不做**那次干预。它只判断「该不该叫」，以及给出一个建议参数，让人来决定。

为什么默认不动手
----------------
施加脉冲会改变她的轨迹。按 HANDOFF §10 的教训，这类事要白默拍板，不能由工具顺手做。
真正执行需要：停 P2 → 改存档 → 启 P2，那是人操作的动作，不是工具自动做的。

用法：
    python tools\prod.py            # 只判断，不动
    python tools\prod.py --json

判据与 tools/vitals.py 一致：`verdict == "frozen"` 或 `"tiny"` 时给出建议。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from tools.vitals import run as vitals  # noqa: E402

# 唤醒实验里 50 步就够（445 区域），200 步更稳（562 区域）。
PULSE_STEPS = 50
PULSE_NOISE = 0.5


def diagnose() -> dict:
    v = vitals()
    verdict = v["verdict"]
    needs = verdict in ("frozen", "tiny")
    return {
        "verdict": verdict,
        "needs_wake": needs,
        "why": {
            "frozen": "她被吸住了：张力 0、离开率 0、max_amp 接近 ceiling。",
            "tiny": "结构太小：边数低于 Drive.min_edges，张力算不出来，动作只能是 HOLD。",
        }.get(verdict, "不需要叫。"),
        "pulse": {"steps": PULSE_STEPS, "noise": PULSE_NOISE} if needs else None,
        "caution": (
            "施加脉冲会改变她的轨迹（一次性干预，不是常驻机制）。"
            "执行需要先停 P2、改存档、再启 P2——人和存档的事，工具不代劳。"
        ) if needs else None,
        "vitals": v,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="should she be woken, and with what")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    r = diagnose()
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    v = r["vitals"]
    print(f"verdict: {r['verdict']}")
    print(f"  {r['why']}")
    if "regions" in v:
        print(f"  regions={v['regions']} edges={v['edges']} (min {v['min_edges']}) "
              f"tensions={v['tensions']} max_amp={v['max_amp']}")
    if r["needs_wake"]:
        p = r["pulse"]
        print(f"\n  建议：一次 {p['steps']} 步、噪声 {p['noise']} 的脉冲。")
        print(f"  {r['caution']}")
        print("\n  人家没有执行。要执行的话，先明确说一次「叫一次」。")
    else:
        print("\n  不用叫。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
