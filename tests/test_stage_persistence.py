r"""阶段跨重启延续的测试。

2026-11-16：P3 每次重启都新建 StageMonitor，`_current` 从 0 起，events.jsonl 里
于是出现 4 -> 0（15:35 mature，16:21 又回到 0）。ARCHITECTURE §8 写的是阶段转换
单向、没有退行机制——那要求它能跨重启成立。

这组测试钉住 round-trip 与「恢复之后不降」。

Run: .\.venv\Scripts\python.exe tests\test_stage_persistence.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L7_lifecycle.stages import StageMonitor  # noqa: E402


def test_round_trip_keeps_stage_and_history() -> None:
    mon = StageMonitor()
    mon._current = 3
    mon._history = [("2026-11-16T10:00:00+00:00", 1), ("2026-11-16T11:00:00+00:00", 3)]

    back = StageMonitor.from_dict(mon.to_dict())
    assert back.current == 3, back.current
    assert back.history == mon.history, (back.history, mon.history)
    print(f"       round-trip kept stage {back.current} and {len(back.history)} history entries")


def test_a_restored_stage_is_not_lowered_by_evaluation() -> None:
    """恢复成 mature 之后，即使当前观测只够 stage 0，也不能降。"""
    mon = StageMonitor.from_dict({"current": 4, "history": [["t0", 4]]})
    before = mon.current
    idx = mon.evaluate("2026-11-16T12:00:00+00:00")
    assert idx == before, (idx, before)
    assert mon.current == 4
    print(f"       restored at 4, one empty evaluate kept it at {mon.current}")


def test_missing_fields_fall_back_to_zero() -> None:
    mon = StageMonitor.from_dict({})
    assert mon.current == 0
    assert mon.history == []
    print("       empty dict -> stage 0, no history")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
