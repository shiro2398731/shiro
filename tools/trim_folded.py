r"""tools/trim_folded.py — 外部记忆（folded.jsonl）太大时，只留最近的。

为什么有它
----------
2026-11-23：`data/memory/folded.jsonl` 长到 35.7 MB / 13,176 行，没有上限。
每行是「被 fold 掉的一个区域」——记的是她的旧结构，不是她当前的状态。

白默说「有别的做吗」——人家来给它加个上限（手动跑，不改进程）。

默认 dry-run；`--apply` 才重写文件（**先备份**）。

用法：
    .\.venv\Scripts\python.exe tools\trim_folded.py              # 只看
    .\.venv\Scripts\python.exe tools\trim_folded.py --keep 5000  # 只看
    .\.venv\Scripts\python.exe tools\trim_folded.py --apply --keep 5000
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEM = ROOT / "data" / "memory" / "folded.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=5000, help="保留最近多少行（默认 5000）")
    ap.add_argument("--apply", action="store_true", help="真的重写（默认只看）")
    args = ap.parse_args()

    if not MEM.exists():
        print(f"没有 {MEM}")
        return 1

    size_mb = MEM.stat().st_size / 1024 / 1024
    lines = MEM.read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(lines)
    print(f"现在: {size_mb:.1f} MB / {n} 行")
    if n <= args.keep:
        print(f"行数 {n} <= {args.keep}，不需要 trim")
        return 0

    kept = lines[-args.keep:]
    new_size_mb = sum(len(x) + 1 for x in kept) / 1024 / 1024
    print(f"保留最近 {args.keep} 行 -> 约 {new_size_mb:.1f} MB（丢掉 {n - args.keep} 行）")

    if not args.apply:
        print("（dry-run，没写；要写加 --apply）")
        return 0

    bak = MEM.with_suffix(MEM.suffix + ".bak")
    shutil.copy2(MEM, bak)
    print(f"备份: {bak.name}")
    MEM.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"完成: {MEM.stat().st_size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
