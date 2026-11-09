r"""M1 接线量的验收：增益有界、方向正确、可 ablation。

增益是调节量，不是生成量：它只能落在 [lo,hi]，且 disabled 时恒为 1.0。

Run: .\.venv\Scripts\python.exe tests\test_adaptive_gain.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.L5_cognition.adaptive_gain import ErrorDrivenGain  # noqa: E402


def test_high_error_raises_gain_low_error_lowers_it() -> None:
    good = ErrorDrivenGain(rate=0.2)
    for _ in range(60):
        good.observe(True)
    bad = ErrorDrivenGain(rate=0.2)
    for _ in range(60):
        bad.observe(False)
    assert good.gain() < 1.0 < bad.gain(), (good.gain(), bad.gain())
    assert good.gain() <= 0.85 and bad.gain() >= 1.35, (good.gain(), bad.gain())
    print(f"       gain: all-hit={good.gain():.3f}  all-miss={bad.gain():.3f}")


def test_gain_is_bounded() -> None:
    g = ErrorDrivenGain(lo=0.8, hi=1.4)
    for _ in range(1000):
        g.observe(False)
        assert 0.8 <= g.gain() <= 1.4
    for _ in range(1000):
        g.observe(True)
        assert 0.8 <= g.gain() <= 1.4


def test_ablation_returns_neutral() -> None:
    g = ErrorDrivenGain(enabled=False)
    for _ in range(100):
        g.observe(False)
    assert g.gain() == 1.0
    print("       ablation: gain pinned at 1.0")


def test_accuracy_matches_observations() -> None:
    g = ErrorDrivenGain()
    seq = [True] * 7 + [False] * 3
    for h in seq:
        g.observe(h)
    assert g.observations == 10 and abs(g.accuracy - 0.7) < 1e-9


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