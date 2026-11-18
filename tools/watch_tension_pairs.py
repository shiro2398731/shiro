"""区域对张力的快照：每 N 分钟记一次「每一对区域的张力」。

为什么：第 88 节说要看「同一个区域对」的张力随时间怎么变，
才能分开「幸存者偏差 / 局部循环 / 真的变紧」三种原因。

observations.jsonl 只记了 tension_sum（总和），没有每对的张力——所以历史没法重建。
从今天开始记。

输出：data/logs/tension_pairs.csv
  列：t, a, b, amount
"""
import csv
import sys
import time
import pathlib

import numpy as np

ROOT = pathlib.Path(r"C:\neolaunchertest\3\bot-3021443116\shiro_alice")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/logs/tension_pairs.csv"
INTERVAL_S = 300   # 5 分钟
TOP_N = 60         # 每次只记最强的 N 对（不然文件太大）


QUANTILES = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)


def snapshot() -> list[dict]:
    """按**分位**取样，不是按 top N。

    2026-11-18 11:10：原来只记最强的 60 个（TOP_N）。但第 90 节查到她的张力是**双峰**的：
      - 第一个峰在 1 附近（57%）；
      - 第二个峰在 10-20（p75-p90）；
      - 长尾到 374。

    **top 60 只覆盖最上面的 9%，中间那个峰（10-20）它一直看不见。**
    而 `per_tension` 涨正是中间层在长。所以 top N 会一直掩盖真变化。

    现在按分位取：p0/p1/p5/p10/p25/p50/p75/p90/p95/p99/p100 各取一个代表。
    **这样「哪个峰在动」能直接看出来。**
    """
    from src.common import persist
    from src.L5_cognition.stream import Stream
    st = Stream()
    if not persist.load_l5(st):
        return []
    sp = st.landscape.state_space
    ts = list(st.drive.tensions(sp, st.graph))
    if not ts:
        return []
    amts = np.asarray([float(t.amount) for t in ts], dtype=float)
    order = np.argsort(amts)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows = []
    for q in QUANTILES:
        idx = int(np.clip(np.searchsorted(amts[order], np.percentile(amts, q)), 0, len(ts) - 1))
        t = ts[int(order[idx])]
        a = getattr(t, "a", None) or getattr(t, "from_region", None)
        b = getattr(t, "b", None) or getattr(t, "to_region", None)
        rows.append({"t": now, "a": a, "b": b,
                     "amount": round(float(t.amount), 4), "q": q})
    return rows


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    new = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", "a", "b", "amount", "q"])
        if new:
            w.writeheader()
        for i in range(rounds):
            rows = snapshot()
            for r in rows:
                w.writerow(r)
            f.flush()
            top = rows[0] if rows else None
            print(f"[{i+1}/{rounds}] {len(rows)} pairs recorded; "
                  f"top = {top}", flush=True)
            if i < rounds - 1:
                time.sleep(INTERVAL_S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
