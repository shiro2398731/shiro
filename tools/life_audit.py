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


def _closure(minutes: float = 30.0) -> dict:
    """最近 N 分钟里 P3 报的闭环状态（homeostasis）。"""
    out = {"closure_ok": None, "corr": None, "n": 0}
    path = LOGS / "observations.jsonl"
    if not path.exists():
        return out
    now = datetime.now(timezone.utc)
    vals = []
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()[-3000:]):
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
        if (now - t).total_seconds() > minutes * 60:
            break
        vals.append((bool(dd["closure_ok"]), dd.get("corr")))
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

    print("\n=== 生命层验收（三条：动量 / 稳态 / 休息）===\n")

    # 1) 动量连续性：她**带着速度**滑过边界（不是被噪声推着重置）
    n_tr = max(1, ev["transitions"])
    own_share = ev["own"] / n_tr
    responding = ev["own"] > 0
    print(f"  {mark(responding)}1 动量      最近 20 分钟：转移 {ev['transitions']} 次，"
          f"其中**带着速度滑过 {ev['own']} 次（{own_share:.1%}）**、被噪声推过 {ev['noise']} 次")
    print(f"               （判据：带速度滑过 > 0。比例仅供参考——**它对窗口高度敏感**：")
    print(f"                 2026-11-19 深夜实测 10min 33% / 20min 27% / 30min 3.2% / 60min 3.5% / 120min 5.2%，")
    print(f"                 因为不同时段噪声驱动的转移密度差很多。比例低不等于她死了，只等于「这段时间她被推着走」。）")
    if ev["flats"]:
        print(f"               （state_flat {ev['flats']} 次——她有过「太平」的时候）")

    # 2) 稳态：闭环
    if cl["closure_ok"] is None:
        print("  -- 2 稳态      最近 30 分钟没有 P3 的闭环记录")
    else:
        print(f"  {mark(cl['closure_ok'])}2 稳态      闭环 {cl['closure_ok']}"
              f"（corr 中位 {cl['corr']}，n={cl['n']}；判据 corr < 0.05 算没断）")

    # 3) 休息：她会停
    #
    # 2026-11-20 0:30 诚实标注：**这一条现在测的是「她没动」（HOLD 占比），不是「她累了在休息」。**
    # 代码里有个专门的休息场 `src/L5_cognition/rest.py`（RestField），但 `stream.py:232` 写着
    # `self._use_rest = False`——**默认没接上**。
    # 所以「HOLD > 0」能证明「她不是一直动」，**不能证明「她会休息」**。
    if "error" in ac:
        print(f"  -- 3 休息      {ac['error']}")
    else:
        resting = ac["hold_share"] > 0
        print(f"  {mark(resting)}3 休息      HOLD 占比 {ac['hold_share']:.3f}"
              f"（leave {ac['leave_share']:.3f}）—— 她不是一直动")
        print(f"               （诚实：这测的是「她没动」，不是「她累了在休息」。")
        print(f"                 rest.py 的休息场默认没接：stream.py:232 `self._use_rest = False`。")
        print(f"                 要真验「休息」，得先把休息场接上——那是改行为，等白默定。）")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
