r"""三阀门审计（2026-11-28，读 shiromo 之后的口径）。

不是模块接口审计；按 shiromo 的三个阀门看：
A 想的门：无外部输入时，她还在不在想（steps/ignition/AST 观测是否继续）？
B 偏差的平：张力分布是不是被压成常量、决策是不是只在极小边距上抖动？
C 主动源：动作/目标由内驱场决定，还是由「最近未去过/回家」规则决定？「说」有没有出口？

只读：读 observations、l5 存档、之前的审计 JSON；不 step、不改状态。

Run: .\.venv\Scripts\python.exe tools\trunk_valve_audit.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

N_MIN = 60.0


def _p2_rows():
    rows = []
    for line in Path("data/logs/observations.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("process") != "P2":
            continue
        try:
            t = datetime.fromisoformat(str(d.get("t")))
        except Exception:
            continue
        rows.append((t, d.get("data") or {}))
    return rows


def _old(dt) -> float:
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0


def main() -> int:
    rows = _p2_rows()
    recent = [(t, d) for t, d in rows if _old(t) <= N_MIN]
    if len(recent) < 2:
        recent = rows[-2:]
    (t0, d0), (t1, d1) = recent[0], recent[-1]
    span = max((t1 - t0).total_seconds() / 60.0, 1e-9)
    ds = int(d1.get("stream_steps", 0)) - int(d0.get("stream_steps", 0))
    di = int(d1.get("ignitions", 0)) - int(d0.get("ignitions", 0))
    dn = int(d1.get("attn_n", 0)) - int(d0.get("attn_n", 0))
    steps_pm = ds / span
    ign_pm = di / span
    ast_pm = dn / span

    st = Stream()
    persist.load_l5(st)
    hist = [float(v) for v in getattr(st, "_tension_history", ()) if np.isfinite(float(v))]
    if hist:
        arr = np.asarray(hist, dtype=float)
        cv = float(arr.std() / max(abs(arr.mean()), 1e-12))
        now = arr[-1]
        med = float(sorted(arr)[len(arr) // 2])
        margin = abs(now - med) / max(abs(now), 1e-12)
    else:
        cv = margin = float("nan")

    p4 = Path("data/logs/P4.heartbeat")
    p4_age = (datetime.now(timezone.utc) - datetime.fromtimestamp(
        p4.stat().st_mtime, timezone.utc)).total_seconds() / 60.0 if p4.exists() else None

    audit = {}
    try:
        audit = json.loads(Path("data/exp/l4_l5_audit.json").read_text(encoding="utf-8"))["summary"]
    except Exception:
        pass
    cos_target = audit.get("cos_drive_target_mean")

    out = {
        "window_min": round(span, 1),
        "steps_per_min": round(steps_pm, 2),
        "ignitions_per_min": round(ign_pm, 2),
        "ast_obs_per_min": round(ast_pm, 2),
        "tension_cv": round(cv, 6) if np.isfinite(cv) else None,
        "decision_margin_rel": round(margin, 6) if np.isfinite(margin) else None,
        "cos_drive_target": cos_target,
        "p4_age_min": None if p4_age is None else round(p4_age, 1),
        # verdicts
        "valve_A_thought_running": bool(steps_pm > 1.0 and ast_pm > 0.01),
        "valve_B_bias_flat": bool((np.isfinite(cv) and cv < 0.005)
                                  and (np.isfinite(margin) and margin < 0.001)),
        "valve_C_world_dominant": bool(cos_target is not None and abs(cos_target) < 0.05),
        "valve_C_speech_offline": bool(p4_age is None or p4_age > 60.0),
    }
    dest = Path("data/exp/trunk_valve_audit.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("== 三阀门审计（近 %.0f 分钟）==" % span)
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"  明细: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
