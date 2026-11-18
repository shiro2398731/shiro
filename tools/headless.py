"""
tools/headless.py

Start, stop and inspect shiro without a terminal being part of her.

A console session must not be a vital organ. This launcher
  * starts the three processes detached, with no console window at all
    (DETACHED_PROCESS | CREATE_NO_WINDOW on Windows; a new session elsewhere),
  * sends their output to files, never to a tty,
  * stops them through the file control channel, i.e. by retiring the epoch,
    so every process stops on its own fence check.

Usage:
    python tools/headless.py start
    python tools/headless.py status
    python tools/headless.py stop
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import control  # noqa: E402
from tools.keeper import below_normal  # noqa: E402

PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():                       # non-Windows layout
    PY = ROOT / ".venv" / "bin" / "python"

MODULES = [
    ("P1", "src.runtime.process_1_fast"),
    ("P3", "src.runtime.process_3_life"),
    ("P2", "src.runtime.process_2_mind"),
]

# 看护（tools/keeper.py）不是她的一部分，是"谁还活着"的守夜人。它必须
#   * 最后起：模块先各自起来，看护只补缺的；
#   * 最先停：否则停她的时候会有东西一边收尸一边重生。
KEEPER_MODULE = "tools.keeper"
LOGS = ROOT / "data" / "logs"
PIDS = ROOT / "data" / "control" / "pids.json"

# Windows: no console, no inheriting handles, own process group.
FLAGS = 0x00000008 | 0x08000000          # DETACHED_PROCESS | CREATE_NO_WINDOW


def _running() -> list[int]:
    """PIDs of live shiro module processes (Windows via CIM, else /proc)."""
    if os.name != "nt":
        out = []
        for p in Path("/proc").iterdir():
            try:
                cmd = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            except Exception:
                continue
            if "src.runtime.process_" in cmd:
                out.append(int(p.name))
        return out
    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -match 'src\\.runtime\\.process_' } | "
         "ForEach-Object { $_.ProcessId }"],
        capture_output=True, text=True)
    return [int(x) for x in ps.stdout.split() if x.isdigit()]


def _keeper_running() -> list[int]:
    """PIDs of live keeper processes."""
    if os.name != "nt":
        out = []
        for p in Path("/proc").iterdir():
            try:
                cmd = (p / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            except Exception:
                continue
            if KEEPER_MODULE in cmd:
                out.append(int(p.name))
        return out
    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -match 'tools[.\\\\]keeper' } | "
         "ForEach-Object { $_.ProcessId }"],
        capture_output=True, text=True)
    return [int(x) for x in ps.stdout.split() if x.isdigit()]


def start() -> int:
    already = _running()
    if already:
        print(f"[headless] already running ({len(already)} processes); stop first")
        return 2
    LOGS.mkdir(parents=True, exist_ok=True)
    pids = {}
    for tag, module in MODULES:
        log = (LOGS / f"{tag}.log").open("ab")
        kwargs = dict(cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                      stdin=subprocess.DEVNULL, close_fds=True)
        if os.name == "nt":
            kwargs["creationflags"] = FLAGS
        else:
            kwargs["start_new_session"] = True
        # 2026-11-18 晚：加 `-u`（unbuffered）。stdout 重定向到文件时 Python 默认块缓冲，
        # 于是「她刚刚做了什么」要等 4-8 KB 才落盘——今天 `reorigin` 就是因此被怀疑没跑。
        # 她不该依赖一个终端来让日志实时出现；`-u` 是这条依赖最便宜的解法。
        proc = subprocess.Popen([str(PY), "-u", "-m", module], **kwargs)
        pids[tag] = proc.pid
        log.close()
        # 降到 BelowNormal：她的步进是常驻负载，这台机器上还有别人的训练在跑。
        # 与 keeper 用的是同一个函数（子进程继承优先级类，设跳板就够）。
        below_normal(proc.pid)
        print(f"[headless] {tag} pid={proc.pid} log=data/logs/{tag}.log")
        time.sleep(0.8)
    PIDS.parent.mkdir(parents=True, exist_ok=True)
    PIDS.write_text(json.dumps(pids, indent=2), encoding="utf-8")

    # 守夜人最后起：它只补缺的，不参与首启。
    if not _keeper_running():
        log = (LOGS / "keeper.log").open("ab")
        kwargs = dict(cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                      stdin=subprocess.DEVNULL, close_fds=True)
        if os.name == "nt":
            kwargs["creationflags"] = FLAGS
        else:
            kwargs["start_new_session"] = True
        kp = subprocess.Popen([str(PY), "-u", "-m", KEEPER_MODULE], **kwargs)
        log.close()
        below_normal(kp.pid)
        pids["KEEPER"] = kp.pid
        PIDS.write_text(json.dumps(pids, indent=2), encoding="utf-8")
        print(f"[headless] KEEPER pid={kp.pid} log=data/logs/keeper.log")
    else:
        print(f"[headless] keeper already running ({len(_keeper_running())} processes)")

    print(f"[headless] started; nothing here needs a console to stay alive")
    return 0


def status(timeout_s: float = 8.0) -> int:
    cid = control.send("status")
    reply = control.wait_reply(cid, timeout_s=timeout_s)
    if reply is None:
        print("[headless] no reply (is P1 running?)")
        return 1
    print(json.dumps(reply.get("result"), ensure_ascii=False, indent=2))
    return 0


def stop(timeout_s: float = 20.0) -> int:
    # 守夜人最先停。它还在的话，下面每一次收尸都会被它当成崩溃、马上重生一个，
    # 于是"停她"变成"看着它们互相拉锯"。
    for pid in _keeper_running():
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True)
            else:
                os.kill(pid, 9)
        except Exception:                                 # noqa: BLE001
            pass
    if not _running():
        # Nothing to stop. Writing a command here would leave it in the inbox
        # for the *next* boot to execute on arrival -- which is exactly how a
        # stale stop killed a fresh run on 2026-11-14.
        print("[headless] nothing running")
        return 0
    cid = control.send("stop")
    reply = control.wait_reply(cid, timeout_s=timeout_s)
    if reply is None:
        print("[headless] stop not acknowledged; processes will still fence on "
              "their own epoch check within a few seconds")
    else:
        print(f"[headless] acknowledged: {reply.get('result')}")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _running():
            print("[headless] all processes gone")
            return 0
        time.sleep(0.5)
    print("[headless] still running; forcing")
    for pid in _running():
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True)
            else:
                os.kill(pid, 9)
        except Exception:
            pass
    return 0 if not _running() else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("start", "stop", "status", "ps"))
    args = ap.parse_args()
    if args.action == "start":
        return start()
    if args.action == "stop":
        return stop()
    if args.action == "status":
        return status()
    print(json.dumps({"pids": _running()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())