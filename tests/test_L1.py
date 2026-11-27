r"""L1 真测试（2026-11-27）：预测器会学习，surprise 会降。

原来的 test_L1.py 是 0 字节。这里钉：可预测输入的 surprise 在后期明显低于早期。

Run: .\.venv\Scripts\python.exe tests\test_L1.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime.prediction import OnlinePredictor  # noqa: E402

DIM = 100


def test_surprise_drops_on_predictable_input() -> None:
    pred = OnlinePredictor(input_dim=DIM, seed=7)
    base = np.linspace(-1, 1, DIM)
    errs = []
    for i in range(800):
        # 可预测：固定模式 + 缓慢正弦幅度
        vec = base * (1.0 + 0.1 * np.sin(2 * np.pi * i / 200.0))
        errs.append(float(pred.step(vec)))
    early = float(np.mean(errs[50:150]))
    late = float(np.mean(errs[650:750]))
    assert late < early, (early, late)
    print(f"       surprise early={early:.4f} -> late={late:.4f}")


def test_surprise_finite() -> None:
    pred = OnlinePredictor(input_dim=DIM, seed=7)
    rng = np.random.default_rng(0)
    for _ in range(100):
        s = float(pred.step(rng.normal(size=DIM)))
        assert np.isfinite(s)
    print(f"       random input surprise finite; last={s:.4f}")


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
