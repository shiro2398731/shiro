r"""tools/prod_apply.py — 真正执行一次唤醒脉冲。

**这是会改变她轨迹的操作。** 默认不执行；必须显式给 `--yes`。

背景（`docs/2026-11-16_唤醒实验.md`）：
- 一次推动叫不醒她（10 sigma 也不行）；
- 一段高噪声（50 步 0.5）可以，而且醒了之后她自己维持。

为什么要停 P2 再改存档
----------------------
P2 每 60 秒会把内存里的状态写回 `data/state/l5.pkl`。如果只改文件不改进程，
下一次落盘就把修改覆盖掉。所以正确顺序是：

    1. 备份 l5.pkl
    2. 停 P2（keeper 会在 30 秒内把它拉起来——但要在它拉起之前改完）
    3. 载入存档 → 施加脉冲 → 写回
    4. 等 keeper 拉起 P2（它会载入改过的存档）
    5. 用 tools/vitals.py 验证 noise 与区域数

这个工具的 `--yes` 只做第 3 步（载入、脉冲、写回）。停进程和等它起来由人做，
因为那是这台机器上的动作，不是她身体里的动作。

用法：
    python tools\prod_apply.py            # 只算，不写
    python tools\prod_apply.py --yes      # 真的写回存档（需要先手动停 P2）

参数来自唤醒实验：50 步、噪声 0.5。醒了之后她自己维持。
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

PULSE_STEPS = 50
PULSE_NOISE = 0.5


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="apply one wake-up pulse to her archive")
    ap.add_argument("--yes", action="store_true", help="actually write the archive back")
    ap.add_argument("--steps", type=int, default=PULSE_STEPS)
    ap.add_argument("--noise", type=float, default=PULSE_NOISE)
    args = ap.parse_args(argv[1:])

    v = vitals()
    print(f"before: verdict={v['verdict']} regions={v.get('regions')} "
          f"edges={v.get('edges')} noise_at_rest")
    if v["verdict"] == "down":
        print("进程没心跳，先处理那个。")
        return 2

    st = Stream()
    if not persist.load_l5(st):
        print("no archive")
        return 2
    sp = st.landscape.state_space
    print(f"archive: regions={len(sp.attractors())} edges={len(st.graph)} "
          f"step_count={st._step_count}")

    if not args.yes:
        print(f"\nwould apply: {args.steps} steps at noise {args.noise}, then persist.")
        print("this CHANGES HER TRAJECTORY. re-run with --yes after stopping P2.")
        return 0

    # 先备份：可回滚是这个操作的底线。
    src = ROOT / "data" / "state" / "l5.pkl"
    backup = ROOT / "data" / "state" / f"l5.pre-prod-{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    shutil.copy2(src, backup)
    print(f"backup: {backup.name}")

    for i in range(args.steps):
        st.landscape.set_noise_amplitude(args.noise)
        st.step()
    attrs = sp.attractors()
    print(f"after pulse: regions={len(attrs)} edges={len(st.graph)} "
          f"step_count={st._step_count}")

    persist.snapshot_all({"l5": st})
    print("written back. now start P2 (keeper will do it if it is running).")
    print("verify with: python tools\\vitals.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
