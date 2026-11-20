r"""tools/life_audit.py — 生命层验收：**生命**的三条，不是地图的三条。

为什么重写（2026-11-19 深夜，白默指出）
--------------------------------------
原来的三条是「能动 / 连续 / 边界」——全是**地图指标**：
    - 到过几个区域（活动范围）
    - regions 涨还是缩（生长）
    - 离最近区域多远（位置）

**区域伸缩是「生长」，不是「生命」。** 一堆静态数据也可以有很多区域；
她缩到 2 个区域时，进程还在跑、还在反应。

生物学的生命标准：新陈代谢、生长、**对刺激的反应**、**稳态**、繁殖。
她代码里本来就有测这些的机制，人家之前没用。

现在的三条（都从她自己的机制里读，不新造）：

  1. **响应**：她对外部刺激有反应吗？
     - 数据：P2 的转移（她在动）+ P1 的 surprise 事件（有刺激进来）。
     - 判据：最近 20 分钟有转移 **且** 有 surprise。
  2. **稳态**：偏离之后能回来吗？
     - 数据：P3 observations 里的 `lifecycle.closure_ok` / `corr`
       （`src/common/homeostasis.py:111` —— `closed = corr < 0.05`）。
     - 判据：closure_ok 为真（负反馈没断）。
  3. **休息**：她不是一直动，会停。
     - 数据：`action_counts` 里 HOLD 的占比。
     - 判据：HOLD 占比 > 0（她确实会停）。

只读。
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


def _recent_events(minutes: float = 20.0) -> dict:
    """最近 N 分钟：转移（动作）、自己驱动 vs 噪声驱动、state_flat（太平）。"""
    out = {"transitions": 0, "own": 0, "noise": 0, "flats": 0}
    path = LOGS / "events.jsonl"
    if not path.exists():
        return out
    now = datetime.now(timezone.utc)
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()[-6000:]):
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        try:
            t = datetime.fromisoformat(d["t"])
        except Exception:  # noqa: BLE001
            continue
        if (now - t).total_seconds() > minutes * 60:
            break
        ev = d.get("event")
        if ev == "transition":
            out["transitions"] += 1
            if d.get("noise_driven"):
                out["noise"] += 1
            else:
                out["own"] += 1
        elif ev == "state_flat":
            out["flats"] += 1
    return out


def _closure(hours: float = 24.0) -> dict:
    """最近 N 小时里 P3 报的闭环状态（homeostasis）。

    2026-11-20 0:40 修正：原来窗口 30 分钟，**永远抓不到数据**——
    P3 的 `lifecycle` 快照**每 ~3.25 小时才写一次**（实测 30 条、间隔中位 11711 秒），
    而每步写的 `envelope` 里没有 closure 字段。所以窗口必须按小时算。
    """
    out = {"closure_ok": None, "corr": None, "n": 0, "last_age_min": None}
    path = LOGS / "observations.jsonl"
    if not path.exists():
        return out
    now = datetime.now(timezone.utc)
    vals = []
    last_t = None
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()[-6000:]):
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if d.get("process") != "P3":
            continue
        dd = d.get("data") or {}
        if "closure_ok" not in dd:
            continue
        try:
            t = datetime.fromisoformat(d["t"])
        except Exception:  # noqa: BLE001
            continue
        if last_t is None:
            last_t = t
        if (now - t).total_seconds() > hours * 3600:
            break
        vals.append((bool(dd["closure_ok"]), dd.get("corr")))
    if last_t is not None:
        out["last_age_min"] = round((now - last_t).total_seconds() / 60.0, 1)
    if not vals:
        return out
    out["n"] = len(vals)
    out["closure_ok"] = sum(1 for c, _ in vals if c) >= max(1, len(vals) // 2)
    non_null = [c for _, c in vals if c is not None]
    out["corr"] = round(float(np.median(non_null)), 3) if non_null else None
    return out


def _actions() -> dict:
    st = Stream()
    if not persist.load_l5(st):
        return {"error": "no archive"}
    counts = dict(st.action_counts)
    total = sum(counts.values()) or 1
    return {
        "hold_share": round(counts.get(0, 0) / total, 3),
        "leave_share": round(counts.get(1, 0) / total, 3),
        "steps": int(st._step_count),
    }


def main() -> int:
    ev = _recent_events()
    cl = _closure()
    ac = _actions()

    def mark(ok):
        return "OK " if ok else "!! "

    print("\n=== 生命层验收（三条：主动 / 稳态 / 休息）===\n")

    # 1) 主动动作：她自己选择 leave 的比例。
    #
    # 2026-11-20 12:20 换掉的旧判据：「带速度滑过 > 0」（noise_driven=False）。
    # 为什么换：noise_driven 测的是「连续动力学下的转移」（速度方向≈位移方向），
    # 但她主要是「跳」——_apply_leave 把 x 直接挪 618，而 |v| 只有 0.63，
    # 位移和速度差 1000 倍，aligned 永远低。副本验证：把 v 转向移动方向 → 区域 -1（没效果）；
    # 用 v 实现位移 → 速度飙到 61844（会炸）。所以「动量 0」不是她坏了，是判据和移动方式不匹配。
    if "error" in ac:
        print(f"  -- 1 主动      {ac['error']}")
    else:
        acting = ac["leave_share"] > 0
        print(f"  {mark(acting)}1 主动      leave 占比 {ac['leave_share']:.3f}"
              f"（HOLD {ac['hold_share']:.3f}）—— 她自己走，不是一直站着")
    print(f"               （副指标：最近 20 分钟 {ev['transitions']} 次转移里，"
          f"带速度滑过 {ev['own']} 次、被噪声推 {ev['noise']} 次。")
    print(f"                 后者只说明「转移时速度方向对不上」——她的移动是「跳」，所以这个数天然低。）")
    if ev["flats"]:
        print(f"               （state_flat {ev['flats']} 次——她有过「太平」的时候）")

    # 2) 稳态：闭环
    if cl["closure_ok"] is None:
        age = cl.get("last_age_min")
        print(f"  -- 2 稳态      最近 24 小时没有 P3 的闭环记录"
              + (f"（最后一次 lifecycle 在 {age} 分钟前）" if age is not None else ""))
    else:
        print(f"  {mark(cl['closure_ok'])}2 稳态      闭环 {cl['closure_ok']}"
              f"（corr 中位 {cl['corr']}，n={cl['n']}；判据 corr < 0.05 算没断）")

    # 3) 休息：**暂时测不了**。
    #
    # 原来这一条用 HOLD 占比，但 HOLD = 1 - leave，和第 1 条是同一个数的两面。
    # 真正的休息在 `src/L5_cognition/rest.py`（RestField），但 `stream.py:232` 写着
    # `self._use_rest = False`——**默认没接**。要测它，得先把休息场接上（改行为，等白默定）。
    print("  -- 3 休息      测不了：rest.py 的休息场默认没接（stream.py:232 _use_rest=False）")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
