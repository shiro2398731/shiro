r"""tools/existence_audit.py — 「存在」这一层的三笔账，一次量完（只读）。

为什么有它
----------
2026-11-18 白默把「存在」拆成三样：**位置不漂、不崩、能动**。
这三样在今天之前都没有一个数，全靠人肉观察。这个工具把三样各自变成一个数：

    A 位置   她的坐标 |x|、她离最近区域多远、地形（mu 范数）多大、最近邻间距、
             以及最重要的比例：一步能走多少 / 到别的区域有多远。
    B 不崩   P2.log 里崩过几次、第一次/最后一次（和 tools/vitals.py 同一判据）。
    C 能动   在候选目标（dist >= s，即 `_apply_leave` 会考虑的那些）里，
             一步最多覆盖到它们的多少距离。

判据怎么读
----------
A 里 `|x|` 单独看没有意义——这个空间是平移不变的，整体搬到哪都一样。
真正有意义的两个数是：
    `最近区域距离 / (4*s)`   —— 她有没有离开区域群（锚的上限是 4*s）
    `s / 到其它区域中位`     —— 一步能覆盖多少（今天实测 0.05%）
所以「位置不漂」应该判的是**地形散布**（mu 范数的分位）有没有无界增长，
不是 |x| 有没有变大。这条是 2026-11-18 下午的结论，写在这里免得下次又去追 |x|。

用法：
    python tools\existence_audit.py                  # 看当前 l5.pkl
    python tools\existence_audit.py --name l5.ALIVE-baseline-20261116_233943.pkl
    python tools\existence_audit.py --sweep          # 把 data/state 里几个关键存档排成表
    python tools\existence_audit.py --json

只读存档与日志；不写任何东西。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

LOGS = ROOT / "data" / "logs"

# 这几个是「位置漂移」这件事上时间不同的存档，sweep 时按这个顺序排。
SWEEP = [
    "l5.ALIVE-baseline-20261116_233943.pkl",   # 11-16 夜间基线（152 区域）
    "l5.pre-explore-20261117_233624.pkl",      # 11-17 探索之前
    "l5.BROKEN-drift-20261118_082458.pkl",     # 11-18 早上漂到撞上限
    "l5.recentered.pkl",                       # 08:23 「重新居中」那一次
    "l5.fixed.pkl",                            # 08:24 修完
    "l5.pkl",                                  # 现在
]


def _crashes() -> dict:
    log = LOGS / "P2.log"
    if not log.exists():
        return {"total": 0, "first": None, "last": None}
    total, first, last = 0, None, None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "崩溃" not in line:
            continue
        total += 1
        first = first or line.strip()
        last = line.strip()
    return {"total": total, "first": first, "last": last}


def _nn_median(mus: np.ndarray, chunk: int = 256) -> float:
    """最近邻间距的中位。分块算，免得 3200x3200x128 一次撑爆。"""
    n = mus.shape[0]
    if n < 2:
        return 0.0
    sq = (mus * mus).sum(axis=1)
    mins = np.empty(n, dtype=float)
    for i in range(0, n, chunk):
        blk = mus[i:i + chunk]
        d2 = sq[i:i + chunk, None] + sq[None, :] - 2.0 * blk @ mus.T
        d2 = np.maximum(d2, 0.0)
        np.fill_diagonal(d2[:, i:i + chunk], np.inf)
        mins[i:i + chunk] = np.sqrt(d2.min(axis=1))
    return float(np.median(mins))


def _visited_hits(st, mus: np.ndarray) -> int:
    """「她去过的地方」能不能对上「区域」。

    2026-11-17 之后 `_visited_places` 存的是**区域 uid（int）**，不是位置 tuple
    （见 stream.py 与 persist.py）。这个函数以前还拿质心 tuple 去对，所以恒为 0——
    2026-11-27 修：按 uid 对；老存档里的 tuple 顺手也按位置对一次。
    """
    visited = set(getattr(st, "_visited_places", None) or ())
    if not visited:
        return 0
    space = st.landscape.state_space
    uids = {int(getattr(a, "uid", -1)) for a in space.attractors()}
    places = {tuple(np.round(np.asarray(m, dtype=float), 4).tolist()) for m in mus}
    by_uid = {int(v) for v in visited if isinstance(v, int)}
    by_pos = set()
    if len(mus):
        dim = len(mus[0])
        for v in visited:
            if isinstance(v, (tuple, list)) and len(v) == dim:
                by_pos.add(tuple(np.round(np.asarray(v, dtype=float), 4).tolist()))
    return len(by_uid & uids) + len(by_pos & places)


def _pair_stats(mus: np.ndarray, chunk: int = 256) -> dict:
    """全区域两两距离的分位。这才是「地形散布」——|x| 是平移不变的，它才不是。"""
    n = mus.shape[0]
    if n < 2:
        return {"n_pairs": 0}
    sq = (mus * mus).sum(axis=1)
    parts = []
    for i in range(0, n, chunk):
        blk = mus[i:i + chunk]
        d2 = sq[i:i + chunk, None] + sq[None, :] - 2.0 * blk @ mus.T
        d2 = np.maximum(d2, 0.0)
        # 只取上三角，避免重复计入
        rows = np.arange(d2.shape[0])[:, None]
        cols = np.arange(d2.shape[1])[None, :]
        keep = cols > (rows + i)
        parts.append(np.sqrt(d2[keep]))
    d = np.concatenate(parts) if parts else np.array([])
    if d.size == 0:
        return {"n_pairs": 0}
    return {
        "n_pairs": int(d.size),
        "pair_p10": float(np.percentile(d, 10)),
        "pair_median": float(np.median(d)),
        "pair_p90": float(np.percentile(d, 90)),
    }


def measure(name: str = "l5.pkl") -> dict:
    st = Stream()
    if not persist.load_l5(st, name=name):
        return {"name": name, "error": "no-archive"}

    sp = st.landscape.state_space
    attrs = sp.attractors()
    if not attrs:
        return {"name": name, "error": "no-regions"}

    mus = np.stack([np.asarray(a.mu, dtype=float) for a in attrs])
    sig = np.asarray([float(a.sigma) for a in attrs], dtype=float)
    x = np.asarray(st._st.x, dtype=float)

    norm_mu = np.linalg.norm(mus, axis=1)
    d_x = np.linalg.norm(mus - x, axis=1)
    s = st._own_scale() or 0.0

    # `_apply_leave` 的候选：除自己以外的区域（当地方案里还要求 dist >= s 的更严筛选，
    # 这里两种都报，免得口径不同又说岔）
    d_others = d_x[d_x > 1e-9]
    cand = d_others[d_others >= s] if s > 0 else d_others

    out = {
        "name": name,
        "regions": len(attrs),
        "edges": len(st.graph),
        "norm_x": float(np.linalg.norm(x)),
        "d_nearest": float(d_x.min()),
        "anchor_limit": float(4.0 * s),
        "nearest_in_anchor": bool(d_x.min() <= 4.0 * s) if s else None,
        "mu_norm_min": float(norm_mu.min()),
        "mu_norm_median": float(np.median(norm_mu)),
        "mu_norm_max": float(norm_mu.max()),
        "nn_median": _nn_median(mus),
        "sigma_median": float(np.median(sig)),
        "s": float(s),
        "mu_norm_cv": float(np.std(norm_mu) / np.median(norm_mu)) if np.median(norm_mu) else None,
        "dist_others_p10": float(np.percentile(d_others, 10)) if d_others.size else None,
        "dist_others_median": float(np.median(d_others)) if d_others.size else None,
        "dist_others_p90": float(np.percentile(d_others, 90)) if d_others.size else None,
        "cand_n": int(cand.size),
        "cand_median": float(np.median(cand)) if cand.size else None,
        "step_over_gap": float(s / float(np.median(cand))) if cand.size and s > 0 else None,
        "visited_places": len(getattr(st, "_visited_places", set()) or ()),
        "visited_hits_places": _visited_hits(st, mus),
        "leave_rate": float(st.leave_share),
    }
    out.update(_pair_stats(mus))
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return "nan"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:,.2f}"
        return f"{v:.4f}"
    return "-" if v is None else str(v)


def show(r: dict) -> None:
    if "error" in r:
        print(f"{r['name']}: {r['error']}")
        return
    print(f"\n=== {r['name']} ===")
    print(f"  regions {r['regions']}  edges {r['edges']}")
    print(f"  |x|            {_fmt(r['norm_x'])}   (平移不变，单看没意义)")
    print(f"  d_nearest      {_fmt(r['d_nearest'])}   (anchor 4s = {_fmt(r['anchor_limit'])}, "
          f"inside={r['nearest_in_anchor']})")
    print(f"  mu 范数        min {_fmt(r['mu_norm_min'])}  "
          f"median {_fmt(r['mu_norm_median'])}  max {_fmt(r['mu_norm_max'])}"
          f"  (cv {_fmt(r['mu_norm_cv'])})")
    print(f"  地形两两距离   p10 {_fmt(r.get('pair_p10'))}  "
          f"median {_fmt(r.get('pair_median'))}  p90 {_fmt(r.get('pair_p90'))}")
    print(f"  nn 间距中位    {_fmt(r['nn_median'])}   sigma 中位 {_fmt(r['sigma_median'])}"
          f"   s {_fmt(r['s'])}")
    print(f"  到其它区域     p10 {_fmt(r['dist_others_p10'])}  "
          f"median {_fmt(r['dist_others_median'])}  p90 {_fmt(r['dist_others_p90'])}")
    print(f"  候选(dist>=s)  n {r['cand_n']}  中位 {_fmt(r['cand_median'])}")
    print(f"  步长/间距      {_fmt(r['step_over_gap'])}"
          + (f"   ({r['step_over_gap'] * 100:.3f}%)" if r["step_over_gap"] else ""))
    print(f"  visited_places {r['visited_places']}   "
          f"对上区域 {r['visited_hits_places']}   leave_rate {_fmt(r['leave_rate'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="l5.pkl")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.sweep:
        rows = []
        for nm in SWEEP:
            p = ROOT / "data" / "state" / nm
            if not p.exists():
                continue
            r = measure(nm)
            rows.append(r)
        for r in rows:
            show(r)
        if a.json:
            print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0

    r = measure(a.name)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        show(r)
        c = _crashes()
        print(f"\n=== 不崩 ===\n  crashes {c['total']}（P2.log，与 vitals 同判据）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
