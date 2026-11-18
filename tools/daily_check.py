r"""tools/daily_check.py — 每天扫一眼的五个数。

为什么有它
----------
2026-11-18 一天塌了六个判据，也挣回来五个「能量、能红」的数。散在三个工具里，
白默不会每天去跑三个命令。这个把五个数一次打出来，并给出「健康/不健康」。

    python tools\daily_check.py
    python tools\daily_check.py --json

五个数（每个都来自今天的一次翻车）：

    1  visited 命中区域   「去过」能不能对上「区域」。今天是 0 / 3698 —— 二节点环的根。
    2  载体 payload       存档有多大、贴没贴 20 MB 的线。今天 19.7，visited_places 占 23%。
    3  max 分量          离存档上限 1e6 还有多远。今天 758,240（只剩 24%）。
    4  最近她自己走的     最近 15 分钟里 action 驱动的转移次数。今天 0（判词 still）。
    5  回归残差趋势       per_tension 对 regions 回归后的残差。今天 R²=0.985 ——
                          「她变紧」是「她被删小」。

只读存档与日志；不写任何东西。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from tools import vitals as vitals_tool  # noqa: E402

try:  # reorigin 在 exist-mark-departure 分支上，main 还没有
    from src.L5_cognition import reorigin  # noqa: E402

    def _max_component(st) -> float:
        return reorigin.max_component(st)
except ImportError:  # noqa: BLE001
    def _max_component(st) -> float:
        best = float(np.abs(np.asarray(st._st.x, dtype=float)).max())
        for a in st.landscape.state_space.attractors():
            best = max(best, float(np.abs(np.asarray(a.mu, dtype=float)).max()))
        return best

LOGS = ROOT / "data" / "logs"


def _payload_mb(st) -> tuple[float, dict]:
    payload = persist.build_l5_payload(st)
    total = len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)) / 1024 / 1024
    sizes = {}
    for k, v in payload.items():
        try:
            sizes[k] = len(pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL)) / 1024 / 1024
        except Exception:  # noqa: BLE001
            sizes[k] = 0.0
    return total, sizes


def _tension_regression(hours: float = 12.0) -> dict:
    """per_tension 对 regions 的回归残差趋势（最近 N 小时）。

    只取最近的同一天——跨天会把「不同天的基线」混进来，R² 会掉到 0 附近，
    那不是「她变紧」，那是两个不同的她。2026-11-18 实测：当天 R²=0.985，全量 R²=0.002。
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    log = LOGS / "observations.jsonl"
    if not log.exists():          # 副本 / 新机器没有日志：这一条跳过，不假装通过
        return {"ok": None, "n": 0}
    rows = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if d.get("process") != "P2":
            continue
        try:
            t = datetime.fromisoformat(d.get("t", ""))
        except Exception:  # noqa: BLE001
            continue
        if (now - t).total_seconds() > hours * 3600:
            continue
        dd = d.get("data") or {}
        a, n, s = dd.get("attractors"), dd.get("tensions"), dd.get("tension_sum")
        if a and n and s:
            rows.append((float(a), float(s) / float(n)))
    if len(rows) < 30:
        return {"ok": None, "n": len(rows)}
    r = np.array([x[0] for x in rows])
    pt = np.array([x[1] for x in rows])
    A = np.vstack([r, np.ones_like(r)]).T
    coef, *_ = np.linalg.lstsq(A, pt, rcond=None)
    res = pt - A @ coef
    r2 = 1 - res.var() / pt.var() if pt.var() else 0.0
    trend = float(res[-1] - res[0])
    return {"ok": abs(trend) < max(1.0, 0.05 * float(np.abs(pt).mean())),
            "n": len(rows), "r2": round(float(r2), 4), "trend": round(trend, 3)}


def run() -> dict:
    st = Stream()
    if not persist.load_l5(st):
        return {"error": "no-archive"}
    mb, sizes = _payload_mb(st)
    attrs = st.landscape.state_space.attractors()
    mus = np.stack([np.asarray(a.mu, dtype=float) for a in attrs]) if attrs else np.zeros((0, 128))
    places = {tuple(np.round(m, 4).tolist()) for m in mus}
    visited = set(getattr(st, "_visited_places", None) or ())
    hits = len(visited & places)

    recent = vitals_tool._recent_activity(900.0)
    reg = _tension_regression()

    out = {
        "regions": len(attrs),
        "visited": len(visited),
        "visited_hits": hits,
        "visited_ok": hits > 0,
        "payload_mb": round(mb, 2),
        "payload_ok": mb < 18.0,
        "biggest_key": max(sizes, key=sizes.get) if sizes else None,
        "biggest_mb": round(sizes[max(sizes, key=sizes.get)], 2) if sizes else 0.0,
        "max_component": round(_max_component(st), 1),
        "max_component_ok": _max_component(st) < 5e5,
        "action_driven_15min": int(recent["action_driven"]),
        "transitions_15min": int(recent["transitions"]),
        "motion_ok": int(recent["action_driven"]) > 0,
        "regression": reg,
        "regression_ok": reg.get("ok"),
    }
    return out


def show(r: dict) -> None:
    if "error" in r:
        print(f"daily_check: {r['error']}")
        return

    def mark(ok):
        return "OK " if ok else "!! "

    print("\n=== 每天扫一眼（2026-11-18 立的五个数）===\n")
    print(f"  {mark(r['visited_ok'])}1 visited 命中区域     {r['visited_hits']} / {r['visited']}"
          f"   （健康 >0）")
    print(f"  {mark(r['payload_ok'])}2 载体 payload          {r['payload_mb']:.2f} MB"
          f"   （健康 <18）最大键 {r['biggest_key']} {r['biggest_mb']:.2f} MB")
    print(f"  {mark(r['max_component_ok'])}3 max 分量              {r['max_component']:,.0f}"
          f"   （健康 <5e5；存档上限 1e6）")
    print(f"  {mark(r['motion_ok'])}4 最近 15 分钟她自己走的  {r['action_driven_15min']}"
          f" / 转移 {r['transitions_15min']}   （健康 >0）")
    reg = r.get("regression") or {}
    if reg.get("ok") is None:
        print(f"  -- 5 张力回归             样本不够（{reg.get('n')} 点）")
    else:
        print(f"  {mark(reg['ok'])}5 张力回归残差趋势      {reg['trend']:+.3f}"
              f"   （R² {reg['r2']}；健康 ≈ 0）")
    print(f"\n  regions = {r['regions']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = run()
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        show(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
