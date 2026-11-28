r"""health_check 分类的测试（2026-11-28）。

Run: .\.venv\Scripts\python.exe tests\test_health_check.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.health_check import classify  # noqa: E402


def test_alive_and_walking() -> None:
    assert classify(1.0, 5, 10.0) == "OK"
    print("       fresh hb + steps -> OK")


def test_heartbeat_dead() -> None:
    assert classify(None, 5, 10.0) == "DEAD"
    assert classify(120.0, 5, 10.0) == "DEAD"
    print("       missing/old heartbeat -> DEAD")


def test_alive_but_stalled() -> None:
    assert classify(2.0, 0, 5000.0) == "STALLED"
    print("       fresh hb + no steps + old obs -> STALLED")


def test_alive_slow_but_recent_obs() -> None:
    assert classify(2.0, 0, 60.0) == "OK"
    print("       no steps in 6s but recent obs -> OK")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
