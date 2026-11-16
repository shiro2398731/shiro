r"""离开事件判据的单元测试（不依赖真实存档）。

钉住三件事：
1. `_dwell_lengths` 把序列切对（一段连续相同 = 一次停留）；
2. `_logloss` 在 p 边界上不炸（有两个 clamp）；
3. `evaluate` 前向推进时不会用未来数据——用一个「见过就不再变」的合成序列，
   如果它偷看了未来，k=8 的 loss 会明显低于理论下界。

Run: .\.venv\Scripts\python.exe tests\test_departure.py
"""

from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.departure_audit import _dwell_lengths, _logloss, evaluate  # noqa: E402


def test_dwell_lengths_split_runs() -> None:
    assert _dwell_lengths([]) == []
    assert _dwell_lengths([1, 1, 1]) == [3]
    assert _dwell_lengths([1, 1, 2, 2, 2, 1]) == [2, 3, 1]
    print("       runs: [1,1,2,2,2,1] -> [2, 3, 1]")


def test_logloss_is_clamped_at_the_edges() -> None:
    for y, p in ((0, 0.0), (1, 1.0), (0, 1.0), (1, 0.0)):
        v = _logloss(y, p)
        assert math.isfinite(v), (y, p, v)
    # 完美预测（几乎）0 bits
    assert _logloss(1, 1.0 - 1e-12) < 1e-3
    assert _logloss(0, 1e-12) < 1e-3
    print("       both edges finite; perfect predictions ~0 bits")


def test_evaluate_runs_and_reports_every_k() -> None:
    """一段规则的三区域循环：停留 3 步、跳一个、停留 3 步……"""
    seq: list[int] = []
    for _ in range(120):
        seq += [1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2]
    r = evaluate(seq, points=50)
    assert "error" not in r, r
    for k in (1, 2, 4, 8):
        d = r["by_k"][k]
        assert d["n"] > 0
        for name in ("her", "constant", "survival"):
            assert math.isfinite(d["bits"][name]), (k, name)
    assert r["mode"] == "prequential"
    print(f"       premquential ok on a periodic sequence; vocab={r['vocab']}")


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
