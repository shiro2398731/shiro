"""
tools/recounter.py — 用她现在的结构重读过去的观测。

"复述"：不是把那段经历再跑一遍，而是把当时的原始观测放到她现在的空间里找位置，
然后看她现在会怎么称呼它。她当时没有索引能力，现在有了 —— 旧数据能读出当时读不出来的东西。

用法
----
    python tools\\recounter.py                 # 读 data/journal/l5.wal
    python tools\\recounter.py --wal <path>
    python tools\\recounter.py --json

只读：不写回她，不改区域，不改指代表。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.recounter import read_observations, recounter  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def run(wal: str = "data/journal/l5.wal") -> dict:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive: run P2 first, or point state_dir at one")
    X = read_observations(wal)
    report = recounter(st, X)
    report["wal"] = str(wal)
    report["wal_observations"] = int(0 if X.size == 0 else X.shape[0])
    report["label_of_first"] = (report["rows"][0]["label"] if report["rows"] else None)
    report.pop("rows", None) if report["n"] > 200 else None
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="reread past observations with her current structure")
    ap.add_argument("--wal", type=str, default="data/journal/l5.wal")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    r = run(args.wal)
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    print(f"wal={r['wal']}  observations={r['wal_observations']}")
    print(f"resolved={r['resolved']}  unresolved={r['unresolved']}  "
          f"distinct regions={r['distinct_regions']}")
    if r["label_counts"]:
        top = sorted(r["label_counts"].items(), key=lambda kv: -kv[1])[:6]
        print("现在的称呼：" + ", ".join(f"{k}={v}" for k, v in top))
    print("说明：resolved = 落在某个区域的一个特征尺度之内；落在外面记为 none，")
    print("      不拿'最近的那个'凑数。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
