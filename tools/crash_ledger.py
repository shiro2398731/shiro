r"""tools/crash_ledger.py — 崩溃账本：她崩过几次、哪几类、几次是放弃重启。

为什么有它
----------
2026-11-18 之前，`vitals` 报的 `crashes: 95` 是错的（只认「崩溃」两个字、还漏了 GBK 段）。
crash-auditor 把真账查清了（153 次 = 126 崩溃行 + 27 放弃重启，6 类），
但那是**一次性的人工考古**。这个工具把它变成**随时能重算的一张表**。

判据（全部来自 P2.log 的原始文本，不依赖时间戳）：

    Traceback 末行（`XxxError: …`）        → 每类多少次
    `[SUPERVISOR] 崩溃`                    → 重启过的异常
    `[SUPERVISOR] 同一错误连续 5 次，放弃重启` → 进程真的退出了
    `[P2] stopped` / `fenced, stopping`    → 会话怎么结束的
    P2.log 里有没有日期                    → 这本账能不能给时间

用法：
    python tools\crash_ledger.py            # 打表
    python tools\crash_ledger.py --json     # 机器可读
    python tools\crash_ledger.py --append    # 追加一行到 data/logs/crash_ledger.jsonl

只读 P2.log；--append 只写 data/logs/crash_ledger.jsonl。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOGS = ROOT / "data" / "logs"
P2_LOG = LOGS / "P2.log"
LEDGER = LOGS / "crash_ledger.jsonl"

# 六类（2026-11-18 crash-auditor 查出来的）。顺序 = 匹配优先级。
CLASSES = [
    ("UnboundLocalError _folded", re.compile(r"UnboundLocalError.*_folded")),
    ("ValueError 质心 >1e6", re.compile(r"ValueError.*(centroid|1e6)")),
    ("ValueError 非有限观测", re.compile(r"ValueError.*non-finite")),
    ("NameError bs", re.compile(r"NameError.*'bs'")),
    ("zmq ZMQError 端口", re.compile(r"ZMQError")),
    ("TypeError deepcopy", re.compile(r"TypeError.*TextIOWrapper")),
]

RE_CRASH = re.compile(r"\[SUPERVISOR\]\s*崩溃")
RE_GIVE_UP = re.compile(r"同一错误连续\s*\d+\s*次，放弃重启")
RE_STOPPED = re.compile(r"\[P2\]\s*stopped")
RE_FENCED = re.compile(r"\[P2\]\s*fenced, stopping")
RE_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _lines() -> list[str]:
    """P2.log 是两种编码的拼盘（UTF-8 + GBK 段）；逐行解，失败换 gbk。"""
    if not P2_LOG.exists():
        return []
    out = []
    for bline in P2_LOG.read_bytes().split(b"\n"):
        if not bline:
            continue
        try:
            out.append(bline.decode("utf-8"))
        except UnicodeDecodeError:
            try:
                out.append(bline.decode("gbk"))
            except UnicodeDecodeError:
                out.append(bline.decode("utf-8", errors="replace"))
    return out


def _classify_segments(lines: list[str]) -> tuple[dict, int, int]:
    """按 Traceback 段分类：每段只看**末行**（`XxxError: …`）。

    2026-11-18 的教训：按行数会重复计（一个 Traceback 里同一个异常名可能出现两三次），
    153 这个数只有「每段一次」才对得上（126 崩溃行 + 27 放弃重启 = 153 段异常）。
    """
    counts = {name: 0 for name, _ in CLASSES}
    other = 0
    segments = 0
    i = 0
    n = len(lines)
    while i < n:
        if "Traceback (most recent call last)" not in lines[i]:
            i += 1
            continue
        segments += 1
        head = lines[i]
        i += 1
        end = None
        while i < n:
            line = lines[i]
            if "Traceback (most recent call last)" in line:
                break
            if re.match(r"^\s*[\w.]*(Error|Exception)\b.*:", line) and "File \"" not in line:
                end = line
                i += 1
                break
            i += 1
        if end is None:
            other += 1
            continue
        hit = False
        for name, pat in CLASSES:
            if pat.search(end) or pat.search(head):
                counts[name] += 1
                hit = True
                break
        if not hit:
            other += 1
    return counts, other, segments


def build() -> dict:
    lines = _lines()
    if not lines:
        return {"error": "no-P2.log"}

    crash_lines = 0
    gave_up = 0
    stopped = 0
    fenced = 0
    dated = 0
    for line in lines:
        if RE_DATE.search(line):
            dated += 1
        if RE_CRASH.search(line):
            crash_lines += 1
        if RE_GIVE_UP.search(line):
            gave_up += 1
        if RE_STOPPED.search(line):
            stopped += 1
        if RE_FENCED.search(line):
            fenced += 1

    counts, other, segments = _classify_segments(lines)
    total = sum(counts.values()) + other
    return {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "lines": len(lines),
        "has_dates": dated > 0,
        "exceptions": total,
        "traceback_segments": segments,
        "crash_lines": crash_lines,
        "gave_up": gave_up,
        "sessions_stopped": stopped,
        "sessions_fenced": fenced,
        "by_class": {k: v for k, v in counts.items() if v},
        "unclassified": other,
    }


def show(r: dict) -> None:
    if "error" in r:
        print(f"crash_ledger: {r['error']}")
        return
    print("\n=== 崩溃账本（P2.log）===\n")
    print(f"  行数            {r['lines']:,}")
    print(f"  有日期时间戳    {'是' if r['has_dates'] else '否（这本账给不了时间）'}")
    print(f"  异常总数        {r['exceptions']}   （账上：崩溃行 {r['crash_lines']} + 放弃重启 {r['gave_up']}"
          f" = {r['crash_lines'] + r['gave_up']}）")
    if r["exceptions"] != r["crash_lines"] + r["gave_up"]:
        print(f"    差 {r['crash_lines'] + r['gave_up'] - r['exceptions']}：有 Traceback 头被并发写坏，"
              f"按段数不到（见 docs/2026-11-18_存在_崩溃记账.md 第 2 节）")
    print(f"    崩溃行        {r['crash_lines']}")
    print(f"    放弃重启      {r['gave_up']}")
    print(f"  Traceback 段    {r['traceback_segments']}")
    print(f"  会话结束        stopped {r['sessions_stopped']} / fenced {r['sessions_fenced']}")
    print("\n  分类：")
    for name, n in sorted(r["by_class"].items(), key=lambda kv: -kv[1]):
        print(f"    {name:28s} {n:4d}")
    if r["unclassified"]:
        print(f"    （未归类）                 {r['unclassified']:4d}")
    print(f"\n  verdict: {'OK：有账' if r['exceptions'] else '没有异常记录'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    r = build()
    if a.append and "error" not in r:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        show(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
