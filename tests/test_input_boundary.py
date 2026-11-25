r"""E1 边界：外部输入只能扰动她，不能设定她。

人类有皮肤、草履虫有细胞膜——「自 / 非自」的边界。她没有膜（状态空间是开放的），
能做实的部分是：**外部信号从 L1 进来，只给 L0 加一段会自己衰减的输入电流，
不保留、不写状态**。这个测试钉住三件事：

1. 同种子的两条轨迹，没有输入时完全相同（确定性）；
2. 一段输入之后轨迹确实分开了（她对外部有反应）；
3. 输入幅度自己衰减，之后偏差回落（不是被设定住了）。

Run: .\.venv\Scripts\python.exe tests\test_input_boundary.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime.process_1_fast import FHNNetwork  # noqa: E402

STEPS = 320
INJECT_AT = 120
WARM = 60


def _run(seed: int, inject_at=None, vector=None) -> tuple[np.ndarray, np.ndarray]:
    net = FHNNetwork(seed=seed)
    fast = []
    amp = []
    for i in range(STEPS):
        if inject_at is not None and i == inject_at and vector is not None:
            net.receive(vector)
        snap = net.step()
        fast.append(float(snap["fast"]))
        amp.append(float(net._ext_amp))
    return np.asarray(fast), np.asarray(amp)


def test_no_input_is_deterministic() -> None:
    a, _ = _run(42)
    b, _ = _run(42)
    assert np.array_equal(a, b), "same seed, same trajectory expected"
    print(f"       two no-input runs identical over {STEPS} steps")


def test_input_perturbs_the_trajectory() -> None:
    base, _ = _run(42)
    fed, _ = _run(42, inject_at=INJECT_AT, vector=np.linspace(-1, 1, 37))
    dev = np.abs(fed - base)
    peak = float(dev[INJECT_AT:INJECT_AT + 40].max())
    assert peak > 1e-6, f"input had no effect (peak {peak:g})"
    print(f"       input at step {INJECT_AT}: peak deviation {peak:.3g}")


def test_input_decays_and_does_not_set_state() -> None:
    base, _ = _run(42)
    fed, amp = _run(42, inject_at=INJECT_AT, vector=np.ones(100))
    dev = np.abs(fed - base)
    early = float(dev[INJECT_AT:INJECT_AT + 30].mean())
    late = float(dev[STEPS - 60:].mean())
    assert late < early, f"deviation did not decay: early {early:g}, late {late:g}"
    # 输入幅度自己掉下去（_ext_decay = 0.98/步）
    assert amp[-1] < 0.05 * amp[INJECT_AT], (
        amp[INJECT_AT], amp[-1])
    print(f"       deviation {early:.3g} -> {late:.3g}; "
          f"input amp {amp[INJECT_AT]:.3g} -> {amp[-1]:.3g}")


def test_any_input_length_is_accepted() -> None:
    """向量维度任意；幅度只由归一化决定，不随长度变化。"""
    for n in (7, 37, 128):
        net = FHNNetwork(seed=42)
        net.receive(np.ones(n))
        a = float(net._ext_amp)
        assert a > 0 and np.isfinite(a), (n, a)
    print("       input lengths 7/37/128 all accepted")


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
