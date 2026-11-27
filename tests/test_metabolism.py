r"""真代谢实验（默认关）的测试（2026-11-27）。

钉：
- 默认关（energy 不变、行为不变）；
- 打开后能量守恒在 [0,1]；
- 能量见底 → 下一步强制 HOLD（歇着）；
- 强制 LEAVE 数步 → 能量下降。

Run: .\.venv\Scripts\python.exe tests\test_metabolism.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.action import HOLD, LEAVE  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def _boot() -> Stream:
    st = Stream()
    assert persist.load_l5(st)
    return st


def test_default_off() -> None:
    st = Stream()
    assert st.use_metabolism is False, st.use_metabolism
    print("       default use_metabolism=False")


def test_energy_bounded() -> None:
    st = _boot()
    st.use_metabolism = True
    for _ in range(30):
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        assert 0.0 <= st.energy <= 1.0, st.energy
    print(f"       energy stays in [0,1] ({st.energy:.3f})")


def test_low_energy_forces_rest() -> None:
    st = _boot()
    st.use_metabolism = True
    st.energy = 0.01  # 见底
    st.landscape.set_noise_amplitude(0.1957)
    st.step()
    assert int(st._action) == HOLD, st._action
    assert st.energy >= 0.01, st.energy
    print(f"       low energy -> HOLD (energy {st.energy:.3f})")


def test_leaving_costs_energy() -> None:
    st = _boot()
    st.use_metabolism = True
    st.metabolism_fresh = 0.0  # 隔离：只量「走」的代价
    st.energy = 0.5
    st.force_action = LEAVE
    for _ in range(10):
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
    assert st.energy < 0.5, st.energy
    print(f"       forced LEAVE drains energy -> {st.energy:.3f}")


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
