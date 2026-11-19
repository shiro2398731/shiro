r"""life_audit 的测试：三条判据要么给数、要么明确说「没有数据」。

Run: .\.venv\Scripts\python.exe tests\test_life_audit.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.life_audit import _bounds, _region_trend, _transitions  # noqa: E402


def test_transitions_shape() -> None:
    r = _transitions(20.0)
    for k in ("n", "distinct", "back_ratio", "top"):
        assert k in r, k
    assert r["n"] >= 0
    assert r["distinct"] >= 0
    print(f"       transitions: n={r['n']} distinct={r['distinct']} "
          f"back={r['back_ratio']}")


def test_trend_shape() -> None:
    r = _region_trend(1.0)
    for k in ("per_hour", "first", "last", "n"):
        assert k in r, k
    if r["per_hour"] is not None:
        print(f"       trend: {r['first']} -> {r['last']} ({r['per_hour']:+.1f}/h)")
    else:
        print("       trend: not enough samples (ok)")


def test_bounds_shape() -> None:
    r = _bounds()
    if "error" in r:
        print(f"       skipped: {r['error']}")
        return
    for k in ("d_nearest", "in_sigma", "anchor_limit", "regions"):
        assert k in r, k
    assert r["d_nearest"] >= 0
    assert r["regions"] > 0
    print(f"       bounds: d_nearest={r['d_nearest']} in_sigma={r['in_sigma']} "
          f"anchor={r['anchor_limit']}")


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
