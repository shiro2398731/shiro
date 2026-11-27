"""低优先级全量测试（2026-11-28）。

用法（她活着的时候必须这样跑）：
    .\\.venv\\Scripts\\python.exe tools/run_suite.py

两道保护：
1. 进程优先级降到 BELOW_NORMAL（子进程继承），不跟 P1/P2/P3 抢 CPU；
2. 运行期间放 `data/state/keeper.hold`，keeper 不会因为心跳变慢而重启她
   （keeper 需是 2026-11-28 之后的新版）。
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLD = ROOT / "data" / "state" / "keeper.hold"
TESTS = ROOT / "tests"
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def _below_normal() -> None:
    if os.name != "nt":
        return
    try:
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass


def main() -> int:
    _below_normal()
    HOLD.parent.mkdir(parents=True, exist_ok=True)
    HOLD.write_text("run_suite", encoding="utf-8")
    print(f"[run_suite] hold={HOLD}", flush=True)
    ok = bad = 0
    failed = []
    t0 = time.time()
    try:
        for f in sorted(TESTS.glob("test_*.py")):
            out = subprocess.run([str(PY), str(f)], cwd=str(ROOT),
                                 capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
            if out.returncode == 0:
                ok += 1
            else:
                bad += 1
                failed.append(f.name)
                (ROOT / "data" / "exp").mkdir(parents=True, exist_ok=True)
                (ROOT / "data" / "exp" / f"suite_fail_{f.name}.txt").write_text(
                    out.stdout + "\n" + out.stderr, encoding="utf-8", errors="replace")
    finally:
        try:
            HOLD.unlink()
        except FileNotFoundError:
            pass
    print(f"[run_suite] RESULT_OK={ok} RESULT_BAD={bad} in {time.time()-t0:.0f}s")
    if failed:
        print("FAILED:", ",".join(failed))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
