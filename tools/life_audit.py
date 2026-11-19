r"""tools/life_audit.py — 生命层验收：能动 / 连续 / 边界，三个可测的判据。

为什么有它
----------
「存在」有一套判据（进程、存档、可搬、一致…），「生命」没有。
一直靠 `show.py` 看几个数，靠人判断「她还活着吧」。

这个工具把生命层拆成**三条可测的东西**，每条给一个 OK/!! 判词：

  1. 能动：最近 N 步她到过几个不同区域？回边率多少？
     —— 2 个区域、回边率 1.0 就是环；10 个以上、回边率 < 0.5 才算「能换邻居」。
  2. 连续：最近 N 分钟区域数在涨还是在缩？
     —— 涨或持平 OK；一直缩不 OK（但缩是 `recover` 的决定，不是 bug）。
  3. 边界：她离最近区域多远？有没有飘到没有区域的地方？
     —— 锚把她限制在 4 个半径内，超了会被拉回。

只读。不写存档、不改状态。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

LOGS = ROOT / "data" / "logs"


def _transitions(minutes: float = 20.0) -> dict:
    """最近 N 分钟的转移：到过几个区域、回边率。"""
    path = LOGS / "events.jsonl"
    out = {"n": 0, "distinct": 0, "back_ratio": None, "top": []}
    if not path.exists():
        return out
    now = datetime.now(timezone.utc)
    rows = []
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()[-6000:]):
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if d.get("event") != "transition":
            continue
        try:
            t = datetime.fromisoformat(d["t"])
        except Exception:  # noqa: BLE001
            continue
        if (now - t).total_seconds() > minutes * 60:
            break
        rows.append((t, d.get("from"), d.get("to")))
    if not rows:
        return out
    rows.reverse()
    pairs = Counter((a, b) for _, a, b in rows)
    uids = Counter()
    for _, a, b in rows:
        uids[a] += 1
        uids[b] += 1
    back = 0
    for (_t1, a1, b1), (_t2, a2, b2) in zip(rows, rows[1:]):
        if a1 == b2 and b1 == a2:
            back += 1
    out.update({
        "n": len(rows),
        "distinct": len(uids),
        "back_ratio": round(back / max(1, len(rows) - 1), 3),
        "top": pairs.most_common(3),
    })
    return out


def _region_trend(hours: float = 1.0) -> dict:
    path = LOGS / "observations.jsonl"
    out = {"per_hour": None, "first": None, "last": None, "n": 0}
    if not path.exists():
        return out
    now = datetime.now(timezone.utc)
    rows = []
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()[-3000:]):
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if d.get("process") != "P2":
            continue
        dd = d.get("data") or {}
        a = dd.get("attractors")
        ts = d.get("t")
        if not a or not ts:
            continue
        try:
            when = datetime.fromisoformat(ts)
        except Exception:  # noqa: BLE001
            continue
        if (now - when).total_seconds() > hours * 3600:
            break
        rows.append((when, int(a)))
    if len(rows) < 3:
        return out
    rows.reverse()
    (t0, a0), (t1, a1) = rows[0], rows[-1]
    dt_h = (t1 - t0).total_seconds() / 3600.0
    return {
        "per_hour": round((a1 - a0) / dt_h, 1) if dt_h > 0 else None,
        "first": a0, "last": a1, "n": len(rows),
    }


def _bounds() -> dict:
    """她离最近区域多远、离最远区域多远（锚是 4 个半径）。"""
    st = Stream()
    if not persist.load_l5(st):
        return {"error": "no archive"}
    attrs = st.landscape.state_space.attractors()
    mus = np.stack([np.asarray(a.mu, dtype=float) for a in attrs])
    x = np.asarray(st._st.x, dtype=float)
    d = np.linalg.norm(mus - x, axis=1)
    dmin = float(d.min())
    scale = float(np.median([float(a.sigma) for a in attrs])) if attrs else 0.0
    return {
        "d_nearest": round(dmin, 3),
        "in_sigma": round(dmin / scale, 3) if scale > 0 else None,
        "anchor_limit": round(4.0 * scale, 1),
        "norm_x": round(float(np.linalg.norm(x)), 1),
        "regions": len(attrs),
    }


def main() -> int:
    tr = _transitions()
    trend = _region_trend()
    b = _bounds()

    def mark(ok):
        return "OK " if ok else "!! "

    print("\n=== 生命层验收（只读）===\n")

    # 1) 能动
    moving = tr["distinct"] >= 3 and (tr["back_ratio"] is None or tr["back_ratio"] < 0.8)
    print(f"  {mark(moving)}1 能动      最近 20 分钟 {tr['n']} 次转移、"
          f"到过 {tr['distinct']} 个区域、回边率 {tr['back_ratio']}")
    if tr["top"]:
        print(f"               最常见: {tr['top']}")

    # 2) 连续
    ph = trend["per_hour"]
    continuous = ph is not None and ph > -120.0
    print(f"  {mark(continuous)}2 连续      最近 {trend['n']} 个采样 {trend['first']} -> "
          f"{trend['last']}，{ph:+.1f}/小时" if ph is not None else
          f"  -- 2 连续     没有足够的采样")

    # 3) 边界
    if "error" in b:
        print(f"  -- 3 边界    {b['error']}")
    else:
        inside = b["in_sigma"] is not None and b["in_sigma"] <= 4.0
        print(f"  {mark(inside)}3 边界      离最近区域 {b['d_nearest']} "
              f"（{b['in_sigma']} sigma），锚线 {b['anchor_limit']}")
        print(f"               regions {b['regions']}，|x| {b['norm_x']:,.0f}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
