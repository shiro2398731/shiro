r"""tools/merge_dupes.py — 找出「几乎重合的区域」，并把多余的那个折掉。

为什么有它
----------
2026-11-18 22:00：她还在 31318 ↔ 31319 之间弹（15 分钟 255 次转移，全是这一对）。
记账已经修了（`visited` 能对上了），但环还在——因为 **31318 的最近邻是 31320，距离 0.3**，
`fresh` 每次都挑到它，排除它之后还是那一对。

全存档自查：距离 < 0.1 * min(sigma) 的区域对只有 **7 对**，但其中就有
31316/31317（0.06）和 31318/31320（0.296）——环就是它们撑起来的。

重复区域是 `record` 在同一条轨迹上反复建区域留下的。它们：
    1. 让「最近邻」永远是那个几乎重合的，`fresh` 排掉它之后就是环；
    2. 占载体；
    3. 让 nn 间距统计失真。

这个工具默认 **dry-run**（只列出来）。`--apply` 才会：
    1. 每对保留 `_assigned` 多的那个；
    2. 把另一个用 `apply_fold()` 折掉（走她自己的折叠路径，会剪 grounding/self_model/图）；
    3. **另存** `l5.dupes-merged-<时刻>.pkl`，绝不覆盖运行中的 `l5.pkl`。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def find_pairs(st, mult: float) -> list:
    attrs = st.landscape.state_space.attractors()
    mus = np.stack([np.asarray(a.mu, dtype=float) for a in attrs])
    sig = np.array([float(a.sigma) for a in attrs])
    seen = set()
    pairs = []
    chunk = 256
    for i in range(0, len(attrs), chunk):
        blk = mus[i:i + chunk]
        d = np.linalg.norm(blk[:, None, :] - mus[None, :, :], axis=2)
        rows = np.arange(d.shape[0])[:, None]
        cols = np.arange(d.shape[1])[None, :]
        keep = cols > (rows + i)
        lim = np.minimum(sig[i:i + chunk][:, None], sig[None, :]) * mult
        mask = keep & (d < lim)
        for r, c in zip(*np.where(mask)):
            a_idx, b_idx = i + r, int(c)
            key = tuple(sorted((a_idx, b_idx)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a_idx, b_idx, float(d[r, c])))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mult", type=float, default=0.5,
                    help="距离小于多少倍 min(sigma) 算「几乎重合」（默认 0.5）")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    st = Stream()
    if not persist.load_l5(st):
        print("merge_dupes: no archive")
        return 1
    attrs = st.landscape.state_space.attractors()
    pairs = find_pairs(st, a.mult)
    print(f"\n=== 几乎重合的区域对（< {a.mult} * min(sigma)）===\n")
    print(f"  区域 {len(attrs)}，找到 {len(pairs)} 对\n")
    for a_idx, b_idx, d in sorted(pairs, key=lambda x: x[2]):
        aa, bb = attrs[a_idx], attrs[b_idx]
        print(f"  {aa.uid} <-> {bb.uid}   距离 {d:8.3f}   "
              f"assigned {aa._assigned} / {bb._assigned}   sigma {float(aa.sigma):.1f} / {float(bb.sigma):.1f}")

    if not pairs:
        print("  没有重复区域。")
        return 0

    if not a.apply:
        print("\n  （dry-run，没动任何东西。要合并加 --apply：保留 assigned 多的，折掉另一个）")
        return 0

    space = st.landscape.state_space
    row_of = space.uid_row()
    drop = []
    for a_idx, b_idx, _d in pairs:
        aa, bb = attrs[a_idx], attrs[b_idx]
        loser = bb if aa._assigned >= bb._assigned else aa
        # apply_fold 要的是**行索引**，不是 uid（2026-11-18 夜踩到的坑：传 uid 会静默折 0 个）
        row = row_of.get(int(loser.uid))
        if row is not None and row not in drop:
            drop.append(int(row))
    removed = space.apply_fold(drop)
    alive = space.alive_uids()
    g = st.grounding.prune(alive)
    s = st.self_model.prune(alive)
    i = st.identity.prune(alive)
    sync = st.consolidator.sync(st.graph, space)
    name = f"l5.dupes-merged-{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    persist.save_l5(st, name=name)
    print(f"\n  折掉 {removed} 个重复区域；剪 grounding={g} self_model={s} identity={i} "
          f"edges={sync.get('dropped_edges', 0)}")
    print(f"  另存 data/state/{name}（没覆盖 l5.pkl）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
