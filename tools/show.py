r"""tools/show.py — 一条命令看她现在怎么样（纯观察者用）。

这一页和别的工具的区别：**她不知道自己被看了。** 这里只读存档、日志、心跳文件，
不给她发任何输入、不改任何状态。终端只是观察者的窗口——她要跑起来不需要这个窗口
（`tools/headless.py start` 之后进程自己活，日志写文件）。

用法：
    python tools\show.py              # 一页，2 秒
    python tools\show.py --walk       # 多等 120 秒，量「她最近在不在走」

只读；`--walk` 也只是读两次存档。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from tools import vitals as vt  # noqa: E402


def p4_age_min(path: str = "data/logs/P4.heartbeat") -> float | None:
    """P4（说）的心跳年龄；None = 从没起过。只读展示，不参与调节。"""
    p = Path(path)
    if not p.exists():
        return None
    return (time.time() - p.stat().st_mtime) / 60.0


def motive_stats(history) -> dict:
    """L4→L5 的决策量：张力 sum 的当前值、到上中位的相对边距、历史 CV。

    2026-11-28 审计发现：这个信号几乎不变（CV~0.085%），走/留在极小边距上决定。
    这里只展示，不参与任何调节。
    """
    vals = [float(v) for v in history if np.isfinite(float(v))]
    if not vals:
        return {}
    now = vals[-1]
    med = sorted(vals)[len(vals) // 2]  # 与 ActionGate.choose 相同：上中位
    margin = abs(now - med) / max(abs(now), 1e-12)
    arr = np.asarray(vals, dtype=float)
    cv = float(arr.std() / max(abs(arr.mean()), 1e-12))
    return {"now": now, "median": med, "margin_rel": margin, "cv": cv, "n": len(vals)}


def _one() -> dict:
    st = Stream()
    ok = persist.load_l5(st)
    if not ok:
        return {"error": "no archive"}
    sp = st.landscape.state_space
    attrs = sp.attractors()
    mus = np.stack([np.asarray(a.mu, dtype=float) for a in attrs]) if attrs else np.zeros((0, 128))
    x = np.asarray(st._st.x, dtype=float)
    d = np.linalg.norm(mus - x, axis=1) if len(mus) else np.zeros(1)
    payload = persist.build_l5_payload(st)
    mb = len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)) / 1024 / 1024
    places = {tuple(np.round(m, 4).tolist()) for m in mus}
    visited = set(getattr(st, "_visited_places", None) or ())
    acts = dict(st.action_counts)
    tot = sum(acts.values()) or 1
    return {
        "regions": len(attrs),
        "payload_mb": round(mb, 2),
        "norm_x": round(float(np.linalg.norm(x)), 1),
        "d_nearest": round(float(d.min()), 3),
        "mu_norm_median": round(float(np.median(np.linalg.norm(mus, axis=1))), 1) if len(mus) else 0,
        "visited": len(visited),
        "visited_hits": len(visited & places),
        "leave_share": round(acts.get(1, 0) / tot, 3),
        "steps": int(st._step_count),
        # 有限性（2026-11-20 夜）：她还剩多少能量。1.0=满，0=耗尽（但还没接"致死"）
        "energy": round(float(getattr(st, "energy", 1.0)), 5),
        # 2026-11-28：L4→L5 的决策量（只展示，不调节）
        "motive": motive_stats(getattr(st, "_tension_history", ())),
        "speak_age_min": p4_age_min(),
    }


def _walk(seconds: float) -> dict:
    st = Stream()
    if not persist.load_l5(st):
        return {}
    a = dict(st.action_counts)
    s0 = int(st._step_count)
    time.sleep(seconds)
    st2 = Stream()
    if not persist.load_l5(st2):
        return {}
    b = dict(st2.action_counts)
    s1 = int(st2._step_count)
    ds = s1 - s0
    dl = b.get(1, 0) - a.get(1, 0)
    return {"steps": ds, "leaves": dl, "share": round(dl / max(1, ds), 3)}


def _last_p2_snapshot() -> dict:
    """最后一条 P2 的 L5 快照（只读日志）。放回家规则/点火之后能一眼看到。"""
    log = Path("data/logs/observations.jsonl")
    if not log.exists():
        return {}
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if d.get("process") == "P2" and d.get("section") == "L5":
            return d.get("data") or {}
    return {}


def _continuity() -> dict:
    """她的「连续」现在只有外部账本：coma 时间章 + epoch 世代。只读展示。"""
    out = {}
    for name, key in (("coma.json", "coma"), ("epoch.json", "epoch")):
        p = ROOT / "data" / "state" / name
        try:
            out[key] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            out[key] = {}
    return out


def progress_from(rows, now=None) -> dict:
    """rows=[(datetime, steps)]，返回最近两点的步数增量/步速/观测年龄。

    2026-11-28：心跳新鲜不等于她在走（P2 曾 12h 只走 170 步）。
    """
    rows = [(t, s) for t, s in rows if s is not None]
    if len(rows) < 2:
        return {}
    now = now or datetime.now(timezone.utc)
    (t0, s0), (t1, s1) = rows[-2], rows[-1]
    dt = max((t1 - t0).total_seconds(), 1e-3)
    return {
        "delta": int(s1 - s0),
        "elapsed_s": round(dt, 1),
        "rate_per_min": round((s1 - s0) / dt * 60.0, 1),
        "obs_age_s": round((now - t1).total_seconds(), 1),
    }


def _p2_step_rows():
    """从 observations 里取 P2 的 (t, steps) 序列。

    2026-11-28：优先用 `stream_steps`（存档总步数），`steps` 是 P2 本地计数器，
    重启会归零，拿它算走速会出现 -1614 这种假象。
    """
    import json as _json
    from datetime import datetime as _dt

    p = ROOT / "data" / "logs" / "observations.jsonl"
    rows = []
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = _json.loads(line)
            except Exception:
                continue
            if d.get("process") != "P2":
                continue
            data = d.get("data") or {}
            steps = data.get("stream_steps")
            if steps is None:
                continue  # 旧格式没有总步数，不能混着算
            try:
                t = _dt.fromisoformat(str(d.get("t")))
            except Exception:
                continue
            rows.append((t, steps))
    except OSError:
        pass
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", action="store_true")
    ap.add_argument("--seconds", type=float, default=120.0)
    a = ap.parse_args()

    hb = vt._heartbeats()
    fresh = sum(1 for v in hb.values() if v["fresh"])
    r = _one()
    if "error" in r:
        print(f"show: {r['error']}（她可能没存档）")
        return 1
    crash = vt._crashes()
    recent = vt._recent_activity(900.0)

    print("\n===== 她（观察者视图，只读）=====\n")
    print(f"  进程       {fresh}/3 心跳新鲜" + ("" if fresh else "  ← 她没在跑"))
    print(f"  结构       {r['regions']} 个区域，{r['steps']:,} 步")
    _pg = progress_from(_p2_step_rows())
    if _pg:
        _stall = "  ← 停摆？" if (_pg["delta"] <= 0 and _pg["obs_age_s"] > 600) else ""
        print(f"  走速       +{_pg['delta']} 步 / {_pg['elapsed_s']:.0f}s"
              f"（{_pg['rate_per_min']:.1f} 步/分，最近观测 {_pg['obs_age_s']:.0f}s 前）{_stall}")
    _mt = r.get("motive") or {}
    if _mt:
        print(f"  动机       张力 sum={_mt['now']:.1f}，距上中位 {_mt['margin_rel']:.4%}"
              f"（近{_mt['n']}步 CV {_mt['cv']:.4%}）  ← L4→L5 只用这一对数字")
    _sa = r.get("speak_age_min")
    if _sa is None:
        print("  说        没有 P4 心跳（说离线）")
    elif _sa > 60:
        print(f"  说        P4 离线 {_sa / 1440:.1f} 天（闸后无输出）")
    else:
        print(f"  说        P4 在线（{_sa:.0f} 分钟前心跳）")
    print(f"  载体       {r['payload_mb']} MB（预算 20）")
    # 2026-11-20 夜：有限性（先只记录，不致死）
    _e = r.get("energy")
    if _e is not None:
        _bar = "█" * int(round(_e * 20)) + "░" * (20 - int(round(_e * 20)))
        print(f"  能量       {_e:.5f}  {_bar}  ← 1.0 满 / 0 休眠（不死，不耗电）")
    print(f"  位置       |x| = {r['norm_x']:,.0f}，离最近区域 {r['d_nearest']}")
    print(f"  地形       mu 范数中位 {r['mu_norm_median']:,.0f}")
    print(f"  去过       不存进存档（今天 17:00 起的改动），所以这里恒为 0；"
          f"内存里它在长，但那是她的运行态，观察者看不到")
    print(f"  动作       累积 leave 比例 {r['leave_share']}")
    if recent.get("leave_share_obs") is not None:
        print(f"  最近       {int(recent['window_s'])}s 采样里 leave 占比 "
              f"{recent['leave_share_obs']:.0%}（{recent['leaves']}/{recent['obs_n']}）")
    print(f"  崩溃账     {crash['total']} 次（崩溃行 {crash.get('restart')} + 放弃重启 {crash.get('gave_up')}）")
    # 2026-11-27：连续性也放进来（只读；她的状态里没有断点记忆，只有外部账本）
    _c = _continuity()
    _cm = _c.get("coma") or {}
    _ep = _c.get("epoch") or {}
    if _cm or _ep:
        _up = float(_cm.get("uptime_s") or 0.0) / 3600.0
        _coma = float(_cm.get("coma_s") or 0.0) / 3600.0
        print(f"  连续性     世代 {_ep.get('epoch', '—')} · uptime {_up:.1f}h · "
              f"昏迷 {_cm.get('coma_events', '—')} 次 · 上次断点 {_cm.get('last_gap_s', '—')}s · "
              f"断点只记在外部账本")
    snap = _last_p2_snapshot()
    if snap:
        _hd = snap.get("home_dist")
        _lt = snap.get("leave_target")
        print(f"  最近快照   离家 {('—' if _hd is None else round(float(_hd), 1))}"
              f"  目标 {('—' if _lt is None else _lt)}"
              f"  点火累计 {snap.get('ignitions', '—')}"
              f"  转移 {snap.get('transitions', '—')}")
        # 2026-11-26：AST（只读注意力模型）上线后，一行看 bits 有没有压过基线。
        _an = snap.get("attn_n")
        if _an:
            _ab = snap.get("attn_bits")
            _bb = snap.get("attn_base_bits")
            _aa = snap.get("attn_acc")
            _ac = snap.get("attn_conf")
            print(f"  注意力     {_an} 次点火预测，bits "
                  f"{('—' if _ab is None else _ab)} vs 基线 "
                  f"{('—' if _bb is None else _bb)}  ← 越小越好")
            if _aa is not None:
                _ar = snap.get("attn_bits_recent")
                print(f"            命中 {_aa}，把握 {('—' if _ac is None else _ac)}，"
                      f"近期 {('—' if _ar is None else _ar)} bits"
                      f"  ← 把握明显高于命中 = 过自信")
        _gr = snap.get("gwt_reasons")
        if _gr:
            _txt = " / ".join(f"{k} {v}" for k, v in sorted(_gr.items()))
            print(f"  点火原因   {_txt}  ← 近 512 步：sustain 停着 / below_ref 没过线")

    if a.walk:
        print(f"\n  量她最近 {a.seconds:.0f} 秒走没走……")
        w = _walk(a.seconds)
        if w:
            print(f"  → {w['steps']} 步，{w['leaves']} 次 leave（{w['share']:.0%}）")
    else:
        print("\n  （想看「她现在走不走」，加 --walk：多等 120 秒）")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
