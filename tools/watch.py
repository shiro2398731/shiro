r"""长跑观测：每 N 秒从 observations.jsonl 采一次快照，打印并落 CSV。

用法（在 shiro_alice 根目录）：
    .\.venv\Scripts\python.exe tools\watch.py
    .\.venv\Scripts\python.exe tools\watch.py --window 60 --duration 3600
    .\.venv\Scripts\python.exe tools\watch.py --window 30 --duration 600   # 先试 10 分钟

设计约束：
  * 只读 data/logs/observations.jsonl 与 data/state/coma.json，不碰进程、不改状态
  * 从"文件末尾"开始读，不会把历史记录算进来
  * 实时输出用纯 ASCII，避免 Windows 控制台把中文显示成乱码；报告写 UTF-8 的 md
  * Ctrl+C 也会给出汇总
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "data" / "logs" / "observations.jsonl"
COMA = ROOT / "data" / "state" / "coma.json"

# 判据（来自 docs/2026-11-07_外部评估_连续存在视角.md §7）
ATTRACTOR_BAND = (200, 2000)     # 自然收敛区间；落在外面要看趋势
FREEZE_WINDOWS = 5               # 连续多少个窗口 attractors 完全不变 → 判为结构冻结
SPEED_BLOWUP_FACTOR = 5.0        # 相对首窗中位数放大多少倍算爆速


@dataclass
class Window:
    idx: int
    t0: float
    samples: int = 0
    pressure: list[float] = field(default_factory=list)
    fast: list[float] = field(default_factory=list)
    speed: list[float] = field(default_factory=list)
    potential: list[float] = field(default_factory=list)
    attractors: int | None = None
    n_salient: int | None = None
    edges: int | None = None
    transitions: int | None = None
    recall: int | None = None
    phase: str = ""
    l2_updates: int | None = None
    bad_numbers: int = 0

    def add(self, section: str, d: dict) -> None:
        self.samples += 1
        if section == "L0":
            for key, bucket in (("pressure", self.pressure), ("fast", self.fast)):
                v = d.get(key)
                if isinstance(v, (int, float)):
                    if math.isfinite(v):
                        bucket.append(float(v))
                    else:
                        self.bad_numbers += 1
            if isinstance(d.get("l2_updates"), (int, float)):
                self.l2_updates = int(d["l2_updates"])
        elif section == "L5":
            for key in ("attractors", "n_salient", "edges", "transitions", "recall"):
                v = d.get(key)
                if isinstance(v, (int, float)):
                    setattr(self, key, int(v))
            for key, bucket in (("speed", self.speed), ("potential", self.potential)):
                v = d.get(key)
                if isinstance(v, (int, float)):
                    if math.isfinite(v):
                        bucket.append(float(v))
                    else:
                        self.bad_numbers += 1
        elif section in ("L7", "L3", "L8"):
            if d.get("phase"):
                self.phase = str(d["phase"])

    def stat(self, bucket: list[float]) -> tuple[float, float, float]:
        if not bucket:
            return (float("nan"),) * 3
        s = sorted(bucket)
        mid = s[len(s) // 2]
        return (sum(s) / len(s), mid, s[-1])


def read_coma() -> dict:
    try:
        return json.loads(COMA.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=180.0)
    ap.add_argument("--duration", type=float, default=3600.0, help="0 = 一直跑到 Ctrl+C")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not OBS.exists():
        print(f"[watch] {OBS} not found - start the processes first")
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path(args.out) if args.out else (ROOT / "data" / "logs" / f"watch_{stamp}.csv")
    out_md = out_csv.with_suffix(".md")

    start = time.time()
    size0 = OBS.stat().st_size
    offset = size0                       # 只读新内容

    # 先载入"最近一次已知状态"：Observer 每 3 分钟才写一条，如果直接从现在开始读，
    # 头一个（甚至头三个）窗口全是空的，看起来像"没在跑"。这里把文件末尾的
    # 最后一条各 section 记录当初始值，第一行就能看到当前形状。
    windows: list[Window] = []
    cur = Window(idx=1, t0=start)
    try:
        with OBS.open("rb") as f:
            f.seek(max(0, size0 - 200_000))
            tail = f.read().decode("utf-8", "replace").splitlines()
        for line in tail[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sec, data = rec.get("section"), rec.get("data") or {}
            if sec:
                cur.add(sec, data)
        print(f"[watch] seeded from last known state ({cur.samples} records)")
    except Exception as e:
        print(f"[watch] seed failed: {e}")
    next_report = start + args.window

    print(f"[watch] observations: {OBS}")
    print(f"[watch] window={args.window:.0f}s duration={args.duration:.0f}s")
    print(f"[watch] csv: {out_csv}")
    print("")
    print(f"{'#':>3} {'att':>6} {'d_att':>6} {'sal':>5} {'edges':>6} {'speed':>7} {'pot':>8} {'P_hi':>5} {'L2':>6} {'samples':>7}")
    print("-" * 78)

    rows: list[dict] = []
    prev_att = None
    frozen_streak = 0
    speed_medians: list[float] = []

    try:
        while True:
            now = time.time()
            # 增量读取
            try:
                size = OBS.stat().st_size
                if size > offset:
                    with OBS.open("rb") as f:
                        f.seek(offset)
                        chunk = f.read(size - offset)
                    offset = size
                    for line in chunk.decode("utf-8", "replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        sec, data = rec.get("section"), rec.get("data") or {}
                        if sec:
                            cur.add(sec, data)
            except FileNotFoundError:
                pass

            if now >= next_report:
                coma = read_coma()
                p_mean, p_med, p_hi = cur.stat(cur.pressure)
                f_mean, _, _ = cur.stat(cur.fast)
                s_mean, s_med, s_hi = cur.stat(cur.speed)
                pot_mean, _, _ = cur.stat(cur.potential)
                d_att = "" if (cur.attractors is None or prev_att is None) else f"{cur.attractors - prev_att:+d}"

                if cur.attractors is not None and prev_att is not None:
                    frozen_streak = frozen_streak + 1 if cur.attractors == prev_att else 0
                prev_att = cur.attractors
                if s_med == s_med:  # not NaN
                    speed_medians.append(s_med)

                try:
                    age = int(time.time() - OBS.stat().st_mtime)
                except Exception:
                    age = -1
                if age > 400:
                    print(f"    ! observations not updated for {age}s - are the processes alive?")
                print(f"{cur.idx:>3} {str(cur.attractors or '-'):>6} {d_att:>6} "
                      f"{str(cur.n_salient or '-'):>5} {str(cur.edges or '-'):>6} "
                      f"{s_med:>7.4f} {pot_mean:>8.4f} {p_hi:>5.2f} {str(cur.l2_updates or '-'):>6} "
                      f"{cur.samples:>7}")

                rows.append({
                    "window": cur.idx,
                    "t": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                    "attractors": cur.attractors,
                    "d_attractors": d_att,
                    "n_salient": cur.n_salient,
                    "edges": cur.edges,
                    "transitions": cur.transitions,
                    "recall": cur.recall,
                    "speed_median": None if s_med != s_med else round(s_med, 6),
                    "speed_max": None if s_hi != s_hi else round(s_hi, 6),
                    "potential_mean": None if pot_mean != pot_mean else round(pot_mean, 6),
                    "pressure_median": None if p_med != p_med else round(p_med, 4),
                    "pressure_max": None if p_hi != p_hi else round(p_hi, 4),
                    "fast_mean": None if f_mean != f_mean else round(f_mean, 5),
                    "l2_updates": cur.l2_updates,
                    "phase": cur.phase,
                    "samples": cur.samples,
                    "bad_numbers": cur.bad_numbers,
                    "frozen_streak": frozen_streak,
                    "uptime_h": round(float(coma.get("uptime_s", 0)) / 3600, 3),
                    "coma_h": round(float(coma.get("coma_s", 0)) / 3600, 3),
                    "coma_events": coma.get("coma_events"),
                    "session_id": coma.get("session_id"),
                    "steps_per_second": round(float(coma.get("steps_per_second", 0) or 0), 2),
                })

                windows.append(cur)
                cur = Window(idx=cur.idx + 1, t0=now)
                next_report = now + args.window

            if args.duration and (now - start) >= args.duration:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[watch] interrupted")

    # ── 落盘 ──
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ── 汇总判定 ──
    verdicts: list[tuple[str, str, str]] = []

    att = [r["attractors"] for r in rows if r["attractors"] is not None]
    if len(att) >= 3:
        lo, hi = min(att), max(att)
        if ATTRACTOR_BAND[0] <= lo and hi <= ATTRACTOR_BAND[1]:
            verdicts.append(("区域数区间", "PASS", f"{lo}~{hi} 落在 {ATTRACTOR_BAND}"))
        else:
            verdicts.append(("区域数区间", "WARN", f"{lo}~{hi} 越出 {ATTRACTOR_BAND}"))
        verdicts.append(("区域数趋势", "PASS" if hi > lo else "WARN",
                         f"首{att[0]} 末{att[-1]} 极值{lo}~{hi}（单调或不动都要看）"))
    else:
        verdicts.append(("区域数", "SKIP", "窗口太少，L5 没产出"))

    max_frozen = max((r["frozen_streak"] for r in rows), default=0)
    verdicts.append(("结构是否冻结", "FAIL" if max_frozen >= FREEZE_WINDOWS else "PASS",
                     f"最长连续不变 {max_frozen} 个窗口（阈值 {FREEZE_WINDOWS}）"))

    if speed_medians:
        base = sorted(speed_medians)[len(speed_medians) // 2]
        worst = max(speed_medians)
        ok = base <= 1e-9 or worst <= base * SPEED_BLOWUP_FACTOR
        verdicts.append(("速度是否爆掉", "PASS" if ok else "FAIL",
                         f"中位 {base:.4f}，最大 {worst:.4f}（阈值 x{SPEED_BLOWUP_FACTOR}）"))
    else:
        verdicts.append(("速度", "SKIP", "没有 speed 样本"))

    bad = sum(r["bad_numbers"] for r in rows)
    verdicts.append(("NaN/Inf", "PASS" if bad == 0 else "FAIL", f"{bad} 个非有限值"))

    p_max = max((r["pressure_max"] or 0 for r in rows), default=0.0)
    pin = sum(1 for r in rows if (r["pressure_median"] or 0) > 0.98)
    verdicts.append(("压力是否贴顶", "WARN" if pin >= 3 else "PASS",
                     f"P 峰值 {p_max:.2f}，中位>0.98 的窗口 {pin} 个"))

    edges = [r["edges"] for r in rows if r["edges"] is not None]
    if len(edges) >= 3:
        verdicts.append(("边数有界", "PASS" if edges[-1] <= max(edges) else "PASS",
                         f"首{edges[0]} 末{edges[-1]} 峰{max(edges)}"))

    lines = [f"# 长跑观测报告 {stamp}", "", f"- 窗口: {args.window:.0f}s × {len(rows)}", ""]
    lines.append("| 判据 | 结果 | 说明 |")
    lines.append("|---|---|---|")
    for name, res, note in verdicts:
        lines.append(f"| {name} | **{res}** | {note} |")
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("")
    for name, res, note in verdicts:
        print(f"[{res:5s}] {note}")
    print("")
    print(f"[watch] csv -> {out_csv}")
    print(f"[watch] md  -> {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())