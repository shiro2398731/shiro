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


def _crashes() -> dict:
    """P2.log 里崩过几次。

    2026-11-18：加这个的原因是 11-17 夜里 P2 崩了 4 次、每次 supervisor 自己重启，
    而 `vitals` 的 verdict 一直是 `alive` —— **没有人知道她崩过**。
    她在重启后十几秒内又站起来，所以心跳永远是新鲜的。

    2026-11-18 下午修了计数（crash-auditor 查出来的两个漏）：
      1. P2.log 是两种编码的拼盘（第 32-1407 行是 GBK），`errors="replace"` 把 GBK 的
         「崩溃」读成乱码，31 条一条都没数到；
      2. 「同一错误连续 5 次，放弃重启」不含「崩溃」二字，但那是进程真的退出，更严重。
    所以逐行按 utf-8 解、失败再按 gbk 解；「崩溃」和「放弃重启」分开数。
    旧口径数到 95，真实是 126 次崩溃行 + 27 次放弃重启 = 153。
    """
    log = LOGS / "P2.log"
    out = {"total": 0, "restart": 0, "gave_up": 0, "last": None}
    if not log.exists():
        return out
    try:
        raw = log.read_bytes()
    except Exception:  # noqa: BLE001
        return out
    for bline in raw.split(b"\n"):
        if not bline:
            continue
        try:
            line = bline.decode("utf-8")
        except UnicodeDecodeError:
            try:
                line = bline.decode("gbk")
            except UnicodeDecodeError:
                continue
        if "崩溃" in line:
            out["restart"] += 1
            out["last"] = line.strip()
        elif "放弃重启" in line:
            out["gave_up"] += 1
            out["last"] = line.strip()
    out["total"] = out["restart"] + out["gave_up"]
    return out


def _recent_activity(window_s: float = 300.0) -> dict:
    """最近 N 秒里她真的在动吗——以及是「自己走的」还是「被噪声推的」。

    2026-11-18 下午加这个的原因：`leave_rate` 是**累积**的（今天还刚进了存档），
    所以她在「一直站」和「一直在走」两个极端上，`vitals` 都可能报 alive。

    2026-11-18 17:35 修了一次：原来只从 `events.jsonl` 的 **transition** 里找
    `noise_driven=False`，于是「她做了 LEAVE、但走一步还在同一个区域里」这种
    最正常的情况被算成「没动」。实测：observations 里 60 个点有 32 次 leave，
    而这里报 0。现在改成**从 observations 的 action 字段数 leave**，
    transition 只当补充（看噪声推了多少次换区域）。
    """
    path = LOGS / "events.jsonl"
    out = {"window_s": window_s, "transitions": 0, "action_driven": 0,
           "leaves": 0, "obs_n": 0, "leave_share_obs": None, "last_event_age_s": None}
    now = time.time()
    from datetime import datetime, timezone

    # 主指标：observations 里最近 window 的 action 分布（比例，不是次数）
    obs = LOGS / "observations.jsonl"
    if obs.exists():
        lines = obs.read_text(encoding="utf-8", errors="replace").splitlines()
        n_obs = 0
        for line in reversed(lines[-2000:]):
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if d.get("process") != "P2":
                continue
            dd = d.get("data") or {}
            ts = d.get("t")
            if not ts:
                continue
            try:
                when = datetime.fromisoformat(ts).timestamp()
            except Exception:  # noqa: BLE001
                continue
            if now - when > window_s:
                break
            if dd.get("action") is None:
                continue
            n_obs += 1
            if dd.get("action") == "leave":
                out["leaves"] += 1
        out["obs_n"] = n_obs
        out["leave_share_obs"] = (round(out["leaves"] / n_obs, 3) if n_obs else None)

    if not path.exists():
        return out
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            f.seek(max(0, size - 262_144))
            tail = f.read().decode("utf-8", errors="replace").splitlines()[1:]
    except Exception:  # noqa: BLE001
        return out
    for line in reversed(tail):
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        ts = d.get("t")
        if not ts:
            continue
        try:
            when = datetime.fromisoformat(ts).timestamp()
        except Exception:  # noqa: BLE001
            continue
        age = now - when
        if out["last_event_age_s"] is None:
            out["last_event_age_s"] = round(age, 1)
        if age > window_s:
            break
        if d.get("event") == "transition":
            out["transitions"] += 1
            if not d.get("noise_driven", True):
                out["action_driven"] += 1
    return out


def _region_trend(hours: float = 1.0) -> dict:
    """最近 N 小时区域数的变化率（每分钟多少个）。

    2026-11-18 夜加：她在缩（-60/h 左右），但这件事以前只能翻 observations 手工算。
    现在一条命令就能看「她还在缩吗、缩多快」。
    """
    from datetime import datetime, timezone
    obs = LOGS / "observations.jsonl"
    out = {"per_hour": None, "n": 0, "first": None, "last": None}
    if not obs.exists():
        return out
    now = datetime.now(timezone.utc)
    rows = []
    for line in reversed(obs.read_text(encoding="utf-8", errors="replace").splitlines()[-3000:]):
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
        "n": len(rows),
        "first": a0,
        "last": a1,
    }


def run() -> dict:
    hb = _heartbeats()
    fresh = sum(1 for v in hb.values() if v["fresh"])

    st = Stream()
    if not persist.load_l5(st):
        return {"verdict": "no-archive", "heartbeats": hb}

    sp = st.landscape.state_space
    attrs = sp.attractors()
    ts = st.drive.tensions(sp, st.graph)
    # 尺度无关的张力强度（2026-11-18 10:10）。sum 会随结构膨胀，这个不会。
    _tsum = sum(t.amount for t in ts) if ts else 0.0
    per_tension = round(_tsum / max(1, len(ts)), 4) if ts else 0.0
    acts = st.action_counts
    tot = sum(acts.values()) or 1
    leave_rate = acts.get(1, 0) / tot
    max_amp = max((float(a.amp) for a in attrs), default=0.0)

    min_edges = int(getattr(st.drive, "min_edges", 64))
    edges = len(st.graph)

    # 2026-11-18 下午：判词要能看见「她在，但一步没走」。
    # `leave_rate` 是累积的（0.70），所以「一直站着」和「一直在走」都会报 alive。
    # 这里看最近 15 分钟的转移里，有多少是她自己（action）走出来的。
    recent_long = _recent_activity(900.0)
    no_own_motion = (recent_long["leaves"] == 0
                     and recent_long["transitions"] > 0)

    if fresh == 0:
        verdict = "down"
    elif edges < min_edges:
        verdict = "tiny"
    elif not ts and max_amp >= 0.9 and leave_rate < 0.02:
        verdict = "frozen"
    elif no_own_motion:
        verdict = "still"
    elif ts and leave_rate > 0.02:
        verdict = "alive"
    else:
        verdict = "quiet"

    crash = _crashes()
    recent = _recent_activity()
    return {
        "verdict": verdict,
        "crashes": crash,
        "recent": recent,
        "recent_long": recent_long,
        "heartbeats": hb,
        "fresh": fresh,
        "regions": len(attrs),
        "edges": edges,
        "min_edges": min_edges,
        "tensions": len(ts),
        "global_tension": round(sum(t.amount for t in ts), 2),
        "per_tension": per_tension,
        "leave_rate": round(leave_rate, 3),
        "max_amp": round(max_amp, 4),
        "ceiling": float(sp.ceiling),
        "folded": len(getattr(sp, "_folded", []) or []),
        "step_count": int(st._step_count),
        "space_steps": int(sp._total_steps),
        "region_trend": _region_trend(),
    }


NOTES = {
    "alive": "有活动：张力和离开率都非零。",
    "quiet": "结构够，但这会儿没什么动静。",
    "frozen": "冻结：结构在、但她被吸住了（张力 0、离开率 0、max_amp 接近 ceiling）。",
    "still": "站着：结构在、有转移，但最近 15 分钟一次都不是她自己走的（全是噪声推的）。",
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
    c = r.get("crashes") or {}
    if c.get("total"):
        print(f"  crashes   : {c['total']} = 崩溃行 {c.get('restart')} + 放弃重启 "
              f"{c.get('gave_up')}（P2.log；旧口径只认「崩溃」二字，数到 95）")
    if "regions" in r:
        print(f"  structure : regions={r['regions']} edges={r['edges']} "
              f"(min_edges={r['min_edges']}) folded={r['folded']}")
        tr = r.get("region_trend") or {}
        if tr.get("per_hour") is not None:
            print(f"  regions趋势: 最近 {tr['n']} 个采样 {tr['first']} -> {tr['last']}，"
                  f"**{tr['per_hour']:+.1f}/小时**")
        print(f"  activity  : tensions={r['tensions']} g_tension={r['global_tension']} "
              f"per_tension={r.get('per_tension')} leave_rate={r['leave_rate']} (累计)")
        rec = r.get("recent") or {}
        if rec.get("last_event_age_s") is not None:
            share = rec.get("leave_share_obs")
            share_s = f"{share:.0%}" if share is not None else "n/a"
            print(f"  recent    : {int(rec['window_s'])}s 内 leave 占比 {share_s} "
                  f"（{rec['leaves']}/{rec.get('obs_n', 0)} 条采样），"
                  f"转移 {rec['transitions']} 次；最后事件 {rec['last_event_age_s']}s 前")
            if rec["transitions"] and share is None:
                print("              ⚠ 这个窗口里没有 observations 采样")
        print(f"  wells     : max_amp={r['max_amp']} / ceiling={r['ceiling']}")
        diff = r["space_steps"] - r["step_count"]
        print(f"  clocks    : stream={r['step_count']} space={r['space_steps']} "
              f"(diff {diff:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
