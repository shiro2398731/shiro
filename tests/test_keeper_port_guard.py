r"""keeper 端口守卫（2026-11-26 19:46 的事故）。

事故：P3 心跳停了 30.6 s，keeper 判定它死了，又起一套；旧 P3 还占着 5563，
新 P3 bind 失败 5 次放弃，还闪出一个窗口。
修法：起进程前先探端口；端口还活着就等它自己恢复。

Run: .\.venv\Scripts\python.exe tests\test_keeper_port_guard.py
"""

from __future__ import annotations

import pathlib
import socket
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import keeper as K  # noqa: E402

# run_suite 会在测试期间放 keeper.hold；这组测试要验端口守卫本身，绕过 hold。
K._held = lambda: False


def test_port_open_detects_listener() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = int(s.getsockname()[1])
    assert K.port_open(port) is True, port
    s.close()
    assert K.port_open(port) is False, port
    print(f"       port_open: listening={port} True, after close False")


def test_watch_waits_when_port_held() -> None:
    started = []
    K.event = lambda *a, **k: None          # no log writes in the test
    K.beat_age = lambda tag, now=None: 999.0
    K.port_alive = lambda tag: True
    K.start_module = lambda tag: started.append(tag) or 12345
    rc = K._watch(["P3"], time.time(), K.Watcher())
    assert rc == 0 and started == [], (rc, started)
    print("       heartbeat stale but port held -> no restart")


def test_watch_starts_when_port_is_free() -> None:
    started = []
    K.event = lambda *a, **k: None
    K.beat_age = lambda tag, now=None: 999.0
    K.port_alive = lambda tag: False
    K.start_module = lambda tag: started.append(tag) or 12345
    rc = K._watch(["P3"], time.time(), K.Watcher())
    assert rc == 0 and started == ["P3"], (rc, started)
    print("       heartbeat stale and port free -> restart once")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
