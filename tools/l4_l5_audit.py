r"""L4→L5 接口审计（2026-11-28）。

只读副本。每步记录 L4 输出的复合量（张力分布、偏转向量、调节场、可存活性）
和 L5 实际用掉的东西（动作、位移、目标），再量它们的耦合/坍缩：

- 张力 `sum` 与 `n`（条数）是否混在一起（长尾分布被求和后，n 会冒充强度）；
- L4 的偏转方向与 L5 的实际位移方向是否一致（余弦）；
- L5 的动作与张力分位的关系（占位：rules 下应≈“高于中位就走”）；
- L5 动作/结果是否反过来改变 L4 的张力（反馈）。

Run: .\.venv\Scripts\python.exe tools\l4_l5_audit.py
输出：data/exp/l4_l5_audit.json + 控制台摘要。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 400
CP = Path("data/state/_audit_l45.pkl")


@dataclass
class Row:
    t: int
    action: int
    tension_used: float
    past_median: float
    n: int
    sum: float
    med: float
    p90: float
    mx: float
    drive_norm: float
    disp: float
    cos_drive_disp: float
    factor_norm: float
    rest_mean: float
    energy: float
    uid: int | None
    target_uid: int | None
    cos_drive_target: float
    dsum: float


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(a @ b / (na * nb))


def run() -> list[Row]:
    shutil.copyfile(Path("data/state/l5.pkl"), CP)
    st = Stream()
    assert persist.load_l5(st, name=CP.name)
    rows: list[Row] = []
    prev_sum = 0.0
    for i in range(STEPS):
        x0 = np.asarray(st._st.x, dtype=float).copy()
        drive = getattr(st.landscape, "_drive", None)
        drive_vec = np.asarray(drive, dtype=float) if drive is not None else np.zeros(1)
        tension_used = float(st._tension_now)
        past = list(st._tension_history)
        # 与 ActionGate.choose 完全一致：排序后取 len//2（偶数时是上中位）
        median = float(sorted(past)[len(past) // 2]) if past else float("nan")
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        disp_vec = np.asarray(st._st.x, dtype=float) - x0
        disp = float(np.linalg.norm(disp_vec))
        s = st.drive.summary()
        reg = st.regulation.last
        target_uid = st._leave_target
        cos_target = 0.0
        if target_uid is not None:
            ts = st.landscape.state_space
            row = ts.uid_row().get(int(target_uid))
            if row is not None:
                mu = np.asarray(ts.attractors()[row].mu, dtype=float)
                cos_target = cosine(drive_vec, mu - x0)
        cur_sum = float(s.get("sum", 0.0))
        rows.append(Row(
            t=i, action=int(st._action), tension_used=tension_used,
            past_median=median, n=int(s.get("n", 0)), sum=cur_sum,
            med=float(s.get("median", 0.0)), p90=float(s.get("p90", 0.0)),
            mx=float(s.get("max", 0.0)),
            drive_norm=float(np.linalg.norm(drive_vec)) if drive_vec.size else 0.0,
            disp=disp, cos_drive_disp=cosine(drive_vec, disp_vec),
            factor_norm=float(np.linalg.norm(reg)) if reg is not None else 0.0,
            rest_mean=float(np.mean([st.rest.value(r) for r in range(len(st.landscape.state_space.attractors()))])),
            energy=float(st.energy),
            uid=st.detector.current_attractor,
            target_uid=target_uid, cos_drive_target=cos_target,
            dsum=cur_sum - prev_sum,
        ))
        prev_sum = cur_sum
    return rows


def summarize(rows: list[Row]) -> dict:
    n = len(rows)
    if not n:
        return {}
    sums = np.array([r.sum for r in rows])
    ns = np.array([r.n for r in rows], dtype=float)
    cos = np.array([r.cos_drive_disp for r in rows])
    disp = np.array([r.disp for r in rows])
    leave = np.array([r.action for r in rows], dtype=float)
    cos_t = np.array([r.cos_drive_target for r in rows if r.target_uid is not None])
    dsum = np.array([r.dsum for r in rows])
    # 规则自检：LEAVE 当且仅当 tension_used > 过去中位
    ok = 0
    tot = 0
    for r in rows:
        if not np.isfinite(r.past_median):
            continue
        tot += 1
        ok += int((r.tension_used > r.past_median) == (r.action == 1))
    # 张力 sum 与条数 n 的线性关系：如果 R² 高，sum 主要在数「有几条边」
    if ns.std() > 1e-9 and sums.std() > 1e-9:
        corr_n_sum = float(np.corrcoef(ns, sums)[0, 1])
        beta = float(np.polyfit(ns, sums, 1)[0])
    else:
        corr_n_sum, beta = 0.0, 0.0
    cos_leave = cos[leave == 1]
    cos_hold = cos[leave == 0]
    margins = np.array([r.tension_used - r.past_median for r in rows
                        if np.isfinite(r.past_median)])
    rel_margins = np.abs(margins) / np.maximum(np.abs(sums[: len(margins)]), 1e-12)
    return {
        "steps": n,
        "leave_share": round(float(leave.mean()), 3),
        "leave_rule_consistency": round(ok / max(tot, 1), 4),
        "tension_sum_mean": round(float(sums.mean()), 3),
        "tension_sum_std": round(float(sums.std()), 2),
        "tension_sum_cv": round(float(sums.std() / max(abs(sums.mean()), 1e-12)), 6),
        "tension_n_mean": round(float(ns.mean()), 3),
        "tension_n_std": round(float(ns.std()), 3),
        "corr_n_sum": round(corr_n_sum, 4),
        "sum_per_edge": round(beta, 4),
        "margin_abs_median": round(float(np.median(np.abs(margins))), 3) if margins.size else None,
        "margin_rel_median": round(float(np.median(rel_margins)), 6) if margins.size else None,
        "margin_rel_p90": round(float(np.percentile(rel_margins, 90)), 6) if margins.size else None,
        "cos_drive_disp_mean": round(float(cos.mean()), 4),
        "cos_drive_disp_abs_median": round(float(np.median(np.abs(cos))), 4),
        "cos_drive_disp_leave_mean": round(float(cos_leave.mean()), 4) if cos_leave.size else None,
        "cos_drive_disp_hold_mean": round(float(cos_hold.mean()), 4) if cos_hold.size else None,
        "cos_drive_target_mean": round(float(cos_t.mean()), 4) if cos_t.size else None,
        "dsum_action_corr": round(float(np.corrcoef(dsum[:-1], leave[1:])[0, 1]), 4)
        if dsum.std() > 1e-9 and leave.std() > 1e-9 else 0.0,
        "disp_mean": round(float(disp.mean()), 2),
        "drive_norm_mean": round(float(np.mean([r.drive_norm for r in rows])), 4),
        "factor_norm_mean": round(float(np.mean([r.factor_norm for r in rows])), 4),
        "rest_mean": round(float(np.mean([r.rest_mean for r in rows])), 4),
        "energy_min": round(float(np.min([r.energy for r in rows])), 4),
    }


def main() -> int:
    rows = run()
    try:
        CP.unlink()
    except FileNotFoundError:
        pass
    out = {
        "summary": summarize(rows),
        "rows": [asdict(r) for r in rows],
    }
    dest = Path("data/exp/l4_l5_audit.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    s = out["summary"]
    print("== L4→L5 审计（副本 %d 步）==" % STEPS)
    for k, v in s.items():
        print(f"  {k}: {v}")
    print(f"  明细: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
