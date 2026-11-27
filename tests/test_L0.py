r"""L0 真测试（2026-11-27）：FHN 群体不是死线。

原来的 test_L0.py 是 0 字节空文件，全量测试里算「通过」但什么都没测。
这里钉三件事：跑得动、真的在振、状态有限。

Run: .\.venv\Scripts\python.exe tests\test_L0.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime.process_1_fast import FHNNetwork  # noqa: E402


def test_runs_and_is_finite() -> None:
    net = FHNNetwork(seed=42)
    for _ in range(200):
        snap = net.step()
    assert np.isfinite(float(snap["fast"])), snap
    print(f"       fast={float(snap['fast']):.4f}")


def test_oscillates() -> None:
    net = FHNNetwork(seed=42)
    fast = []
    for _ in range(1500):
        fast.append(float(net.step()["fast"]))
    fast = np.asarray(fast)
    # 群体活动在 [0,1] 来回摆；不要求过零（fast 是活动率，不是电压）
    assert float(fast.std()) > 0.1, fast.std()
    assert float(fast.max() - fast.min()) > 0.5, (fast.min(), fast.max())
    print(f"       range=[{fast.min():.3f},{fast.max():.3f}], std={fast.std():.3f}")


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
