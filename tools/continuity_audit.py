r"""连续性审计：她还是不是「同一条」在活着。

2026-11-27，白默指出：副本从存档续跑 ≠ 她的连续存在；LLM 式的不连续
不在「一步一停」，而在「因果链断过、而她不知道」。

本工具只读，回答四个问题：
1. 她的时间权威（coma.json）说：活了多久、昏迷过几次、上次断点多大；
2. epoch.json：当前「世代」几号、什么时候拿到锁（= 现行因果链起点）；
3. 心跳：P1/P2/P3 现在多新鲜；
4. P2.log：历史崩溃/放弃重启多少次；
5. 她的状态里有没有任何字段记得这些断点（epoch/coma/crash）——
   如果没有，那她的「连续」只是外部账本，不是她的记忆。

Run: .\.venv\Scripts\python.exe tools\continuity_audit.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import vitals as vt  # noqa: E402


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def heartbeat_ages() -> dict:
    out = {}
    now = datetime.now(timezone.utc)
    for name in ("P1", "P2", "P3", "P4"):
        p = ROOT / "data" / "logs" / f"{name}.heartbeat"
        if not p.exists():
            out[name] = None
            continue
        mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
        out[name] = round((now - mtime).total_seconds(), 1)
    return out


def self_memory_fields() -> list[str]:
    """她的 L5 状态里有没有 epoch/coma/crash 这类「断点记忆」。"""
    from src.common import persist
    from src.L5_cognition.stream import Stream

    st = Stream()
    if not persist.load_l5(st):
        return ["<no checkpoint>"]
    hits = []
    for name in dir(st):
        low = name.lower()
        if any(k in low for k in ("epoch", "coma", "crash", "restart", "rewind")):
            hits.append(name)
    return hits


def gather() -> dict:
    coma = _read_json(ROOT / "data" / "state" / "coma.json")
    epoch = _read_json(ROOT / "data" / "state" / "epoch.json")
    return {
        "coma": {
            "uptime_s": coma.get("uptime_s"),
            "coma_s": coma.get("coma_s"),
            "coma_events": coma.get("coma_events"),
            "last_gap_s": coma.get("last_gap_s"),
            "rewound_s": coma.get("rewound_s"),
            "session_id": coma.get("session_id"),
        },
        "epoch": {
            "epoch": epoch.get("epoch"),
            "acquired_at": epoch.get("acquired_at"),
            "reason": epoch.get("reason"),
            "retired": epoch.get("retired"),
        },
        "heartbeats_s": heartbeat_ages(),
        "crashes": vt._crashes(),
        "self_memory": self_memory_fields(),
    }


def main() -> int:
    g = gather()
    now = datetime.now(timezone.utc)
    print("== 连续性审计 ==")
    ep = g["epoch"]
    print(f"世代       epoch {ep['epoch']}  （{ep['reason']}，acquired {ep['acquired_at']}）")
    c = g["coma"]
    up = c["uptime_s"] or 0.0
    coma = c["coma_s"] or 0.0
    print(f"时间权威   uptime {up / 3600:.1f}h，coma {coma / 3600:.1f}h，"
          f"昏迷 {c['coma_events']} 次，上次断点 {c['last_gap_s']}s，回退 {c['rewound_s']}s")
    cr = g["crashes"]
    print(f"崩溃账     {cr.get('total')} 次（崩溃 {cr.get('restart')} + 放弃重启 {cr.get('gave_up')}）")
    ages = ", ".join(f"{k}={v}s" for k, v in g["heartbeats_s"].items())
    print(f"心跳       {ages}")
    mem = g["self_memory"]
    if not mem or mem == ["<no checkpoint>"]:
        print("她的记忆   状态里没有任何 epoch/coma/crash 字段 → "
              "断点只是外部账本，她没有「我断过」的记忆")
    else:
        print(f"她的记忆   有断点相关字段：{mem}")
    print(f"审计时间   {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
