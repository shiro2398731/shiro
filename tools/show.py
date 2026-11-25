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
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from tools import vitals as vt  # noqa: E402


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
    snap = _last_p2_snapshot()
    if snap:
        _hd = snap.get("home_dist")
        _lt = snap.get("leave_target")
        print(f"  最近快照   离家 {('—' if _hd is None else round(float(_hd), 1))}"
              f"  目标 {('—' if _lt is None else _lt)}"
              f"  点火累计 {snap.get('ignitions', '—')}"
              f"  转移 {snap.get('transitions', '—')}")

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
