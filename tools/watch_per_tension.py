"""观察任务：每 10 分钟量一次 per_tension，写进 CSV。

不是为了「看数据」——是为了看它是「真涨」还是「瞬态」。
第 83 节说：如果 3.5%/10 分钟的速率持续，她会在两天内绷断。
这个脚本记录它到底怎么走。

用法：python watch_per_tension.py [轮数]
  默认 10 轮（100 分钟）。
"""
import csv
import sys
import time
import pathlib

ROOT = pathlib.Path(r"C:\neolaunchertest\3\bot-3021443116\shiro_alice")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data/logs/per_tension.csv"
INTERVAL_S = 600   # 10 分钟


def sample() -> dict:
    from src.common import persist
    from src.L5_cognition.stream import Stream
    st = Stream()
    if not persist.load_l5(st):
        return {"t": "", "per_tension": None, "sum": None, "n": None, "regions": None}
    sp = st.landscape.state_space
    ts = st.drive.tensions(sp, st.graph)
    total = sum(t.amount for t in ts) if ts else 0.0
    return {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_tension": round(total / max(1, len(ts)), 4) if ts else 0.0,
        "sum": round(total, 2),
        "n": len(ts),
        "regions": len(sp.attractors()),
    }


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    new = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t", "per_tension", "sum", "n", "regions"])
        if new:
            w.writeheader()
        for i in range(rounds):
            row = sample()
            w.writerow(row)
            f.flush()
            print(f"[{i+1}/{rounds}] {row['t']}  per_tension={row['per_tension']}  "
                  f"sum={row['sum']}  n={row['n']}  regions={row['regions']}", flush=True)
            if i < rounds - 1:
                time.sleep(INTERVAL_S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
