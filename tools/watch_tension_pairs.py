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

ROOT = pathlib.Path(r"C:\neolaunchertest\3\bot-3021443116\shiro_alice")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/logs/tension_pairs.csv"
INTERVAL_S = 300   # 5 分钟
TOP_N = 60         # 每次只记最强的 N 对（不然文件太大）


def snapshot() -> list[dict]:
    from src.common import persist
    from src.L5_cognition.stream import Stream
    st = Stream()
    if not persist.load_l5(st):
        return []
    sp = st.landscape.state_space
    ts = sorted(st.drive.tensions(sp, st.graph), key=lambda t: -t.amount)[:TOP_N]
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows = []
    for t in ts:
        a = getattr(t, "a", None) or getattr(t, "from_region", None)
        b = getattr(t, "b", None) or getattr(t, "to_region", None)
        rows.append({"t": now, "a": a, "b": b, "amount": round(float(t.amount), 4)})
    return rows


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    new = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", "a", "b", "amount"])
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
