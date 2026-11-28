r"""她到底在不在走（2026-11-28）。

事故背景：P2 心跳新鲜、端口在听，但 12 小时只走了 170 步——「活着但停摆」。
心跳只能证明进程在转，不能证明她在走。这个工具同时看三件事：

1. 心跳年龄（进程活着吗）
2. 6 秒内 step_count 有没有涨（她在走吗）
3. 最近一条 P2 观测多久以前（她在记吗）

用法：.\.venv\Scripts\python.exe tools/health_check.py [--sleep 6]
退出码：0=OK，2=STALLED，3=DEAD。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def classify(heartbeat_age: float | None, step_delta: int,
             obs_age_s: float | None, stale_s: float = 600.0) -> str:
    """纯分类函数，方便单测。"""
    if heartbeat_age is None or heartbeat_age > 60.0:
        return "DEAD"
    if step_delta <= 0 and (obs_age_s is None or obs_age_s > stale_s):
        return "STALLED"
    return "OK"


def _steps() -> int | None:
    st = Stream()
    if not persist.load_l5(st):
        return None
    return int(st._step_count)


def _obs_age(process: str = "P2") -> float | None:
    p = ROOT / "data" / "logs" / "observations.jsonl"
    last = None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("process") == process:
            last = d.get("t")
    if not last:
        return None
    try:
        t = datetime.fromisoformat(str(last))
    except Exception:
        return None
    return (datetime.now(timezone.utc) - t).total_seconds()


def heartbeat_age(proc: str) -> float | None:
    p = ROOT / "data" / "logs" / f"{proc}.heartbeat"
    if not p.exists():
        return None
    return time.time() - p.stat().st_mtime


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=6.0)
    a = ap.parse_args(argv)
    hb = {p: heartbeat_age(p) for p in ("P1", "P2", "P3")}
    s0 = _steps()
    time.sleep(a.sleep)
    s1 = _steps()
    delta = 0 if s0 is None or s1 is None else s1 - s0
    oa = _obs_age("P2")
    print("== health ==")
    for p, v in hb.items():
        print(f"  {p} heartbeat: " + ("—" if v is None else f"{v:.1f}s"))
    print(f"  steps: {s0} -> {s1} (+{delta} in {a.sleep:.0f}s)")
    print(f"  last P2 observation: " + ("—" if oa is None else f"{oa:.0f}s ago"))
    verdict = classify(hb.get("P2"), delta, oa)
    print(f"  verdict: {verdict}")
    return {"OK": 0, "STALLED": 2, "DEAD": 3}[verdict]


if __name__ == "__main__":
    sys.exit(main())
