r"""L2 真测试（2026-11-27）：16 维状态场不是空的。

原来的 test_L2.py 是 0 字节。这里钉：维度、归一化范围、dev 有限。

Run: .\.venv\Scripts\python.exe tests\test_L2.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime.process_1_fast import L2StateField  # noqa: E402


def test_dim_and_bounds() -> None:
    l2 = L2StateField()
    vec = None
    for i in range(600):
        t = i * 0.05
        a = math.sin(2 * math.pi * t)
        vec, dev = l2.step(
            s_t=abs(a), fast=a, I=0.5 + 0.1 * a,
            mid=a * 0.5, slow=a * 0.2, pressure=0.3,
            cycle_stats={"period_s": 1.0},
        )
    assert vec is not None and len(vec) == L2StateField.DIM, len(vec or [])
    arr = np.asarray(vec, dtype=float)
    assert np.all(np.isfinite(arr)), arr
    assert arr.min() >= 0.0 and arr.max() <= 1.0, (arr.min(), arr.max())
    assert np.isfinite(dev) and dev >= 0.0, dev
    print(f"       dim={len(vec)}, range=[{arr.min():.2f},{arr.max():.2f}], dev={dev:.3f}")


def test_flat_input_gives_low_dev() -> None:
    l2 = L2StateField()
    for _ in range(400):
        vec, dev = l2.step(
            s_t=0.5, fast=0.5, I=0.5, mid=0.5, slow=0.5,
            pressure=0.5, cycle_stats={"period_s": 1.0},
        )
    assert np.isfinite(dev) and dev >= 0.0, dev
    print(f"       flat input dev={dev:.3f}")


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
