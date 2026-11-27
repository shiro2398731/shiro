r"""架构真实性守卫（2026-11-28）。

白默指出「预测与惊讶居然单独作为一层」后，我们拆了 L1 假分层、删了空壳、
改了文档。这个测试钉住「文档说的 = 代码有的」：

- ARCHITECTURE 不再说「九层」/把 L1 预测当层；
- src 下没有 0 字节非 __init__ 模块；
- 预测器在 runtime/prediction.py，process_1_fast 里旧类名不存在；
- L1_prediction / L2_affect / drives 目录不存在。

Run: .\.venv\Scripts\python.exe tests\test_architecture_truth.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_architecture_says_functional_domains() -> None:
    text = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "## 1. 九层架构" not in text, "ARCHITECTURE 仍有九层架构章节"
    assert "不是九层" in text, "ARCHITECTURE 缺少修正后的口径"
    assert "L1 预测与惊讶" not in text, "ARCHITECTURE 仍把预测当一层"
    assert "功能域" in text, "ARCHITECTURE 缺少功能域口径"
    print("       ARCHITECTURE uses functional domains, not nine layers")


def test_no_empty_modules_under_src() -> None:
    empty = [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*.py")
             if p.stat().st_size == 0 and p.name != "__init__.py"]
    assert not empty, empty
    print("       no 0-byte non-__init__ modules under src")


def test_fake_layer_dirs_gone() -> None:
    for d in ("L1_prediction", "L2_affect"):
        assert not (ROOT / "src" / d).exists(), f"src/{d} 还在"
    assert not (ROOT / "src" / "L4_drive" / "drives").exists()
    print("       L1_prediction / L2_affect / drives gone")


def test_prediction_lives_in_runtime() -> None:
    from src.runtime import process_1_fast
    from src.runtime.prediction import OnlinePredictor

    assert hasattr(process_1_fast, "OnlinePredictor")
    assert not hasattr(process_1_fast, "L1Predictor")
    p = OnlinePredictor(input_dim=8, seed=1)
    import numpy as np
    assert np.isfinite(float(p.step(np.zeros(8))))
    print("       prediction lives in runtime/prediction.py")


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
