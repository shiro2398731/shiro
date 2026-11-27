r"""假分层守卫（2026-11-27）：预测不该是一个「层」。

白默指出：预测与惊讶单独作为一层是错的；L1_prediction 目录是五个 0 字节空壳，
真实现住在 process_1_fast。这个测试钉住重构后的状态：

- `src/L1_prediction/` 不存在；
- 预测器在 `src/runtime/prediction.py`，类名 OnlinePredictor；
- 旧名字 L1Predictor 不再存在（防止命名倒退）。

Run: .\.venv\Scripts\python.exe tests\test_no_fake_layers.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_l1_dir_removed() -> None:
    assert not (ROOT / "src" / "L1_prediction").exists(), "L1_prediction 应该已删除"
    print("       src/L1_prediction removed")


def test_prediction_module_works() -> None:
    from src.runtime.prediction import OnlinePredictor

    p = OnlinePredictor(input_dim=100, seed=7)
    s = float(p.step(np.zeros(100)))
    assert np.isfinite(s) and s >= 0.0, s
    print(f"       OnlinePredictor works, surprise={s:.3f}")


def test_old_name_gone() -> None:
    from src.runtime import process_1_fast

    assert hasattr(process_1_fast, "OnlinePredictor")
    assert not hasattr(process_1_fast, "L1Predictor"), "旧类名 L1Predictor 还在"
    print("       old class name L1Predictor gone")


def test_no_zero_byte_modules() -> None:
    empty = [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")
             if p.stat().st_size == 0 and p.name != "__init__.py"]
    assert not empty, empty
    print("       src has no 0-byte non-__init__ modules")


def test_l2_and_drives_dirs_gone() -> None:
    assert not (ROOT / "src" / "L2_affect").exists(), "L2_affect 应已删除"
    assert not (ROOT / "src" / "L4_drive" / "drives").exists(), "drives/ 应已删除"
    assert not (ROOT / "src" / "main.py").exists(), "空 main.py 应已删除"
    print("       L2_affect / drives / main.py gone")


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
