r"""卡住检测的边界测试。

2026-11-19 深夜：`_arrival_hist` 的判据是
    `len(set(hist)) <= 3 and len(hist) >= 4`
没有自己的测试——只有真机数据。这里把边界钉住。

Run: .\.venv\Scripts\python.exe tests\test_stuck_detection.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 判据（和 stream.py 里那句保持一致；改那边要同步改这里）
HIST_MAXLEN = 6
MAX_DISTINCT = 3
MIN_LEN = 4


def _stuck(hist: list[int]) -> bool:
    return len(set(hist)) <= MAX_DISTINCT and len(hist) >= MIN_LEN


def test_alternating_two_places_is_stuck() -> None:
    """A,B,A,B,... —— 第一版「连续同一个」在这里永不触发，第二版应该判定卡住。"""
    for hist in ([1, 2, 1, 2], [1, 2, 1, 2, 1, 2]):
        assert _stuck(hist) is True, hist
    print("       A->B 交替 -> 判卡住（这就是第一版漏掉的）")


def test_one_place_is_stuck() -> None:
    assert _stuck([1, 1, 1, 1]) is True
    print("       原地不动 -> 判卡住")


def test_three_places_is_stuck() -> None:
    """<=3 的边界：三个地方也算打转。"""
    assert _stuck([1, 2, 3, 1]) is True
    assert _stuck([1, 2, 3, 2]) is True
    print("       三个地方 -> 判卡住（边界）")


def test_four_places_is_not_stuck() -> None:
    """4 个不同地方 -> 不卡（她真的在走）。"""
    assert _stuck([1, 2, 3, 4]) is False
    assert _stuck([1, 2, 3, 4, 5, 6]) is False
    print("       四个以上地方 -> 不判卡住")


def test_short_history_is_not_stuck() -> None:
    """历史不足 4 次 -> 不判（证据不够）。"""
    assert _stuck([1]) is False
    assert _stuck([1, 2]) is False
    assert _stuck([1, 2, 3]) is False
    print("       历史 <4 次 -> 不判卡住")


def test_hist_maxlen_is_six() -> None:
    """和 stream.py 的 deque(maxlen=6) 一致。"""
    from src.L5_cognition.stream import Stream

    st = Stream()
    assert st._arrival_hist.maxlen == HIST_MAXLEN, st._arrival_hist.maxlen
    print(f"       deque maxlen = {st._arrival_hist.maxlen}")


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
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
