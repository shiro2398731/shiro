r"""tools/terrain_apply.py — 改她的地形容量（ceiling），并对现有区域同比例缩放。

**这会改变她的动力学。** 默认 dry-run，必须 `--yes`。

背景（HANDOFF §32）：用她真实的 feature_hist（噪声 0.2）在副本上扫 ceiling：

    ceiling 1.0  ->  7 区域 / 6 边 / 张力 0
    ceiling 0.25 -> 17 / 16 / 0
    ceiling 0.05 -> 82 / 81 / 15
    ceiling 0.02 -> 383 / 382 / 96（离开率 39–40%）
    ceiling 0.01 -> 539 / 538 / 115

`ceiling` 是「单个区域的深度容量」，势阱深度 ∝ amp，梯度 ∝ amp/sigma²。
1.0 时梯度 346，噪声 0.2 是它的 0.06%，推不动；0.02 时梯度 ~7，0.2 就够。

用法：
    python tools\terrain_apply.py                # 只算，不写
    python tools\terrain_apply.py --yes --ceiling 0.02
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from tools.vitals import run as vitals  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="change her terrain capacity")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--ceiling", type=float, default=0.02)
    args = ap.parse_args(argv[1:])

    v = vitals()
    print(f"before: verdict={v['verdict']} regions={v.get('regions')} "
          f"edges={v.get('edges')}")

    st = Stream()
    if not persist.load_l5(st):
        print("no archive")
        return 2
    sp = st.landscape.state_space
    old = float(sp.ceiling)
    print(f"ceiling: {old} -> {args.ceiling}   (ratio {args.ceiling / old:.4f})")
    print(f"regions: {len(sp.attractors())}, amps will scale by the same ratio")

    if not args.yes:
        print("\nwould rewrite the archive. this CHANGES HER DYNAMICS.")
        print("re-run with --yes after stopping P2.")
        return 0

    src = ROOT / "data" / "state" / "l5.pkl"
    backup = ROOT / "data" / "state" / f"l5.pre-terrain-{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    shutil.copy2(src, backup)
    print(f"backup: {backup.name}")

    ratio = args.ceiling / old
    sp.ceiling = float(args.ceiling)
    for a in sp.attractors():
        a.amp = float(a.amp) * ratio
    attrs = sp.attractors()
    if attrs:
        mx = max(float(a.amp) for a in attrs)
        print(f"after scale: {len(attrs)} regions, max_amp={mx:.5f}")

    persist.snapshot_all({"l5": st})
    print("written back. start P2 (keeper will if running).")
    print("verify with: python tools\\vitals.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
