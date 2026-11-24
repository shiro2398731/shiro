"""
tools/keeper.py — 进程之外的看护：谁的模块停了，就按同样的方式把它拉起来。

为什么需要这一层
----------------
每个模块自己有 `_supervised()`（异常退避重启），但那只接得住**抛异常**。
2026-11-15 实测：`kill -9` 掉 P2 的进程树之后，**40 秒内没有任何东西把它拉起来**——
P1 继续走步、`step_count` 一直在涨，而 `P2.heartbeat` 冻在 `06:18:51`。
也就是：她还在，她的思维没了，而且不会自己回来。

进程内重启接不住"进程被外面杀掉"这一类，因为它需要一个还活着的宿主。
所以需要一层在进程之外的看护。它只做一件事：按心跳判断哪个模块停了，再起一个。

判据（不是参数）
----------------
"停了" = 心跳文件的时间戳比 `--stale-s` 更旧。**不看进程表**：
venv 的 `python.exe` 是跳板进程，一个模块对应两个进程（跳板 + 真解释器），
跳板活着不代表模块活着；按进程表数会数出"三个模块六个进程"这种假象。

放弃
----
同一个模块连续 `--fail-limit` 次"刚起来、还没活过 `--min-life-s` 就又停了"→ 记录并退出 1。
理由与 P1 的 `MAX_IDENTICAL_FAILURES` 相同：永久错误（缺文件、端口被占）上无限重启
只会刷日志、把 CPU 烧在退避上，对无人看守的运行比停下更糟。

用法
----
    python tools/keeper.py --once          # 走一拍就退出（手动补一个模块用）
    python tools/keeper.py                 # 常驻，默认每 5 s 一拍
    python tools/keeper.py --tag P2        # 只看护一个模块

退出码：0 = 正常（包括"没有要做的"），1 = 放弃（某个模块熔断）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "data" / "logs"

PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():                       # non-Windows layout
    PY = ROOT / ".venv" / "bin" / "python"

# 顺序与 headless.py 一致：P1 先起（它负责领取 epoch 与总线），最后起 P2。
MODULES = {
    "P1": "src.runtime.process_1_fast",
    "P3": "src.runtime.process_3_life",
    "P2": "src.runtime.process_2_mind",
}

# 与 headless.py 相同的启动方式：没有控制台、不继承句柄、独立进程组。
FLAGS = 0x00000008 | 0x08000000           # DETACHED_PROCESS | CREATE_NO_WINDOW

EVENTS = LOGS / "keeper.jsonl"


def heartbeat_path(tag: str) -> Path:
    return LOGS / f"{tag}.heartbeat"


def beat_age(tag: str, now: Optional[float] = None) -> Optional[float]:
    """秒数：心跳文件有多旧。没有文件返回 None（= 从来没起过）。"""
    p = heartbeat_path(tag)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return None
    return max((time.time() if now is None else now) - mtime, 0.0)


def below_normal(pid: int) -> None:
    """把新进程降到 BelowNormal。子进程继承优先级类，所以设跳板就够。

    为什么要降：这台机器上还有别人的训练在跑（实测 `vlmdisaster` 的 train.py 占 1.8 GB 内存、
    六位数 CPU 秒）。她的步进是常驻负载，不该和别人的任务抢调度。
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(0x0200, False, pid)      # PROCESS_SET_INFORMATION
        if handle:
            k32.SetPriorityClass(handle, 0x00004000)      # BELOW_NORMAL_PRIORITY_CLASS
            k32.CloseHandle(handle)
    except Exception:                                     # noqa: BLE001
        pass                                              # 降不下来不是停机的理由


def start_module(tag: str) -> int:
    """按 headless 的方式起一个模块，返回 Popen 的 pid。"""
    LOGS.mkdir(parents=True, exist_ok=True)
    module = MODULES[tag]
    log = (LOGS / f"{tag}.log").open("ab")
    kwargs = dict(cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                  stdin=subprocess.DEVNULL, close_fds=True)
    if os.name == "nt":
        kwargs["creationflags"] = FLAGS
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen([str(PY), "-m", module], **kwargs)
    log.close()
    below_normal(proc.pid)
    return int(proc.pid)


def event(kind: str, **fields) -> None:
    """一行一个事件，给下一个看日志的人（或下一轮的我）。"""
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind}
    rec.update(fields)
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        with EVENTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass                                  # 写不进事件日志不该让看护停下


class Watcher:
    """决策部分，与进程无关，所以可以单独测。

    每拍调一次 `plan()` 拿到要起的模块，起了就 `note_started()`。
    "刚起来还没心跳"与"真的又死了"必须分开，否则一拍一杀会把同一模块起成一片。
    """

    def __init__(self, stale_s: float = 30.0, min_life_s: float = 60.0,
                 fail_limit: int = 5) -> None:
        self.stale_s = float(stale_s)
        self.min_life_s = float(min_life_s)
        self.fail_limit = int(fail_limit)
        self.last_start: dict[str, float] = {}
        self.fails: dict[str, int] = {}
        self.given_up: dict[str, str] = {}

    def alive(self, tag: str, now: float) -> bool:
        age = beat_age(tag, now)
        return age is not None and age <= self.stale_s

    def waiting(self, tag: str, now: float) -> bool:
        """刚起过、还在等它第一次心跳 —— 这一拍不该再起一个。"""
        t0 = self.last_start.get(tag)
        return t0 is not None and (now - t0) < self.min_life_s

    def plan(self, tags: list[str], now: float) -> list[str]:
        out = []
        for tag in tags:
            if tag in self.given_up:
                continue
            if self.alive(tag, now) or self.waiting(tag, now):
                continue
            out.append(tag)
        return out

    def note_started(self, tag: str, now: float) -> int:
        self.last_start[tag] = now
        self.fails[tag] = self.fails.get(tag, 0) + 1
        return self.fails[tag]

    def note_alive(self, tag: str) -> None:
        """它活下来了：失败计数归零（下一次死是新的一次，不是第 N 次）。"""
        self.fails[tag] = 0
        self.last_start.pop(tag, None)

    def should_give_up(self, tag: str, now: float) -> Optional[str]:
        """刚起的那个又没活过 min_life_s 就停了 → 判定为永久错误。

        只有在"已经起过、并且过了 min_life_s 窗口"的情况下才算数：
        否则第一拍就会把任何一个冷启动中的模块判死。
        """
        t0 = self.last_start.get(tag)
        if t0 is None:
            return None
        if (now - t0) < self.min_life_s:
            return None
        if self.alive(tag, now):
            return None
        if self.fails.get(tag, 0) >= self.fail_limit:
            return (f"{tag} 连续 {self.fails[tag]} 次起来后 {self.min_life_s:.0f}s 内又停了")
        return None


def _watch(tags: list[str], now: float, w: Watcher) -> int:
    """一拍：检查、记录存活、起死掉的、判定熔断。返回给出的退出码（0 表示继续）。"""
    for tag in tags:
        if tag in w.given_up:
            continue
        if w.alive(tag, now):
            if w.fails.get(tag):
                event("recovered", tag=tag, fails=w.fails[tag],
                      age=round(beat_age(tag, now) or 0.0, 1))
            w.note_alive(tag)
            continue

        why = w.should_give_up(tag, now)
        if why:
            w.given_up[tag] = why
            event("gave_up", tag=tag, reason=why, fails=w.fails.get(tag, 0))
            print(f"[keeper] 放弃：{why}", flush=True)
            continue

        if not w.waiting(tag, now):
            age = beat_age(tag, now)
            try:
                pid = start_module(tag)
            except Exception as exc:                      # noqa: BLE001
                event("start_failed", tag=tag, error=f"{type(exc).__name__}: {exc}")
                print(f"[keeper] 起 {tag} 失败：{exc}", flush=True)
                continue
            fails = w.note_started(tag, now)
            event("started", tag=tag, pid=pid, fails=fails,
                  stale_s=None if age is None else round(age, 1))
            print(f"[keeper] {tag} 不在（心跳 {('无' if age is None else f'{age:.0f}s 前')}），"
                  f"已起 pid={pid}（第 {fails} 次）", flush=True)

    if w.given_up and len(w.given_up) == len(tags):
        return 1
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="watch shiro's modules and restart the dead ones")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--stale-s", type=float, default=30.0)
    # 2026-11-24 晚：默认 15 -> 60。`headless.py` 起 keeper 时**从来没传这个参数**，
    # 所以「改成 60」一直没生效——keeper 用 15 秒窗口，冷启动还没心跳就被判死，补一套，
    # 两套抢端口。60 秒是「冷启动 + 第一次心跳」的余量。
    ap.add_argument("--min-life-s", type=float, default=60.0)
    ap.add_argument("--fail-limit", type=int, default=5)
    ap.add_argument("--tag", action="append", choices=sorted(MODULES))
    ap.add_argument("--once", action="store_true", help="走一拍就退出")
    args = ap.parse_args(argv)

    tags = args.tag or ["P1", "P3", "P2"]
    w = Watcher(stale_s=args.stale_s, min_life_s=args.min_life_s,
                fail_limit=args.fail_limit)
    event("boot", tags=tags, interval=args.interval, stale_s=args.stale_s,
          min_life_s=args.min_life_s, fail_limit=args.fail_limit)

    while True:
        code = _watch(tags, time.time(), w)
        if code:
            return code
        if args.once:
            return 0
        time.sleep(max(args.interval, 0.5))


if __name__ == "__main__":
    sys.exit(main())
