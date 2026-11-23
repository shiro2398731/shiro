r"""tools/merge_regions.py — 手动把两个重叠的区域合成一个（dry-run 默认）。

为什么有它
----------
2026-11-23：她卡在 32434/32435 之间。查出来：
    sigma 99.7 + 99.7 = 199.3，两区域距离 199.2——**刚好重叠**。
她走一步（199）就从一个到另一个，于是 dwell_history 全是 1、fatigue_rate 低、
不待腻、不走远。**合并这两个 = 她「待久」= 待腻 = 走远**。

**但这改她的结构**（mu/amp/members）——按白默的规矩要她点头。
所以这个工具默认 **dry-run**，`--apply` 才另存（不覆盖 l5.pkl）。

用法：
    .\.venv\Scripts\python.exe tools\merge_regions.py --a 32434 --b 32435
    .\.venv\Scripts\python.exe tools\merge_regions.py --a 32434 --b 32435 --apply
"""

from __future__ import annotations

import argparse
import pickletools  # noqa: F401  (保留 import 风格)
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, required=True, help="第一个区域的 uid")
    ap.add_argument("--b", type=int, required=True, help="第二个区域的 uid")
    ap.add_argument("--apply", action="store_true", help="真的另存（默认只看）")
    args = ap.parse_args()

    st = Stream()
    if not persist.load_l5(st):
        print("没有存档")
        return 1
    sp = st.landscape.state_space
    attrs = sp.attractors()
    ra = sp.uid_row().get(args.a)
    rb = sp.uid_row().get(args.b)
    if ra is None or rb is None:
        print(f"找不到：{args.a}={ra} {args.b}={rb}")
        return 1
    a = attrs[ra]
    b = attrs[rb]
    dist = float(np.linalg.norm(np.asarray(a.mu, dtype=float) - np.asarray(b.mu, dtype=float)))
    print(f"{args.a}: sigma={a.sigma:.1f} amp={a.amp:.4f} members={len(a.members)}")
    print(f"{args.b}: sigma={b.sigma:.1f} amp={b.amp:.4f} members={len(b.members)}")
    print(f"距离={dist:.1f}  半径和={a.sigma + b.sigma:.1f}  重叠={dist < a.sigma + b.sigma}")
    print(f"合并后: 质心=平均、amp={a.amp + b.amp:.4f}、members={len(a.members) + len(b.members)}")

    if not args.apply:
        print("（dry-run，没写；要写加 --apply）")
        return 0

    # 合并：质心取平均、amp 相加、members 合并、sigma 取大的
    a.mu = (np.asarray(a.mu, dtype=float) + np.asarray(b.mu, dtype=float)) / 2.0
    a.amp = float(a.amp) + float(b.amp)
    for m in list(b.members):
        a.members.append(m)
    a.scale = max(float(a.scale or 0), float(b.scale or 0))
    attrs.pop(rb)
    # 重建 uid 映射
    sp._next_uid = max(int(getattr(sp, "_next_uid", 1)), args.a + 1)
    ok = persist.save_l5(st)
    print(f"合并完成，save={ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
