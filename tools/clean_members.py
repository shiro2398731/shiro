r"""tools/clean_members.py — 她的「成员记忆」里有多少是漂移前的旧坐标？

为什么有它
----------
2026-11-18 晚：查「保存为什么撞 1e6」时发现，`attractors[0].mu` 的最大分量只有 18.7 万，
但它的 `members[0]` 最大分量是 51 万。再一查：**10,095 个 members 里 7,529 个（75%）
离自己的区域质心超过 1e4**，分量全部 > 5e5，全局最大 831,278。

它们是 11-17 那次漂移之前留下的旧坐标——区域质心跟着她搬了，成员没有。
两个后果：
    1. `assert_l5_finite` 也查 members → 保存撞 1e6 → 进程退出（153 次崩溃里的一类）；
    2. 载体 62% 是 members → 折叠按预算砍她的区域。

这个工具**只读**：量清楚有多少、离多远，并且给出「如果按某个阈值清，会清掉多少」。
默认 dry-run；`--apply` 只另存新文件（**不覆盖 l5.pkl**，她还在跑）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

LIMIT = 1e6  # SAVE_MAGNITUDE_LIMIT


def audit(mult: float) -> dict:
    st = Stream()
    if not persist.load_l5(st):
        return {"error": "no archive"}
    attrs = st.landscape.state_space.attractors()
    dists = []
    total = 0
    over = 0
    big = 0
    worst = 0.0
    for a in attrs:
        mu = np.asarray(a.mu, dtype=float)
        sig = float(a.sigma) if a.sigma else 0.0
        for m in a.members:
            total += 1
            v = np.asarray(m, dtype=float)
            d = float(np.linalg.norm(v - mu))
            dists.append(d)
            worst = max(worst, float(np.abs(v).max()))
            if float(np.abs(v).max()) > LIMIT:
                over += 1
            if sig > 0 and d > mult * sig:
                big += 1
    if not total:
        return {"regions": len(attrs), "members": 0}
    arr = np.array(dists)
    return {
        "regions": len(attrs),
        "members": total,
        "dist_p50": round(float(np.median(arr)), 1),
        "dist_p90": round(float(np.percentile(arr, 90)), 1),
        "dist_max": round(float(arr.max()), 1),
        "over_limit": over,
        "over_mult": big,
        "mult": mult,
        "worst_component": round(worst, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=float, default=100.0,
                    help="离质心超过多少个 sigma 算「旧坐标」（默认 100）")
    ap.add_argument("--apply", action="store_true",
                    help="另存 l5.members-pruned-<时刻>.pkl（绝不覆盖 l5.pkl）")
    a = ap.parse_args()

    r = audit(a.mult)
    if "error" in r:
        print(f"clean_members: {r['error']}")
        return 1
    print("\n=== 成员记忆审计（只读）===\n")
    print(f"  区域          {r['regions']}")
    print(f"  成员总数      {r['members']}")
    if not r["members"]:
        return 0
    print(f"  离质心距离    p50 {r['dist_p50']:,.0f} / p90 {r['dist_p90']:,.0f} / max {r['dist_max']:,.0f}")
    print(f"  超过存档上限  {r['over_limit']} 个（分量 > 1e6）")
    print(f"  超过 {int(r['mult'])} sigma   {r['over_mult']} 个（= 可能的旧坐标）")
    print(f"  全局最大分量  {r['worst_component']:,.0f}")

    if a.apply:
        import time
        st = Stream()
        assert persist.load_l5(st)
        removed = 0
        for at in st.landscape.state_space.attractors():
            mu = np.asarray(at.mu, dtype=float)
            sig = float(at.sigma) if at.sigma else 0.0
            if sig <= 0:
                continue
            keep = type(at.members)(
                (m for m in at.members
                 if float(np.linalg.norm(np.asarray(m, dtype=float) - mu)) <= a.mult * sig),
                maxlen=getattr(at.members, "maxlen", 64),
            )
            removed += len(at.members) - len(keep)
            at.members = keep
        name = f"l5.members-pruned-{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        persist.save_l5(st, name=name)
        print(f"\n  清理 {removed} 个旧成员，另存 data/state/{name}（没覆盖 l5.pkl）")
    else:
        print("\n  （dry-run，没写任何东西；要另存就加 --apply）")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
