r"""记忆回路：预演的结果要回到她自己身上。

这条回路是：预演（沙盒）→ 与她实际走到的位置比较 → 命中/未命中进一个独立增益
→ 与一步预测的增益相乘 → 缩放噪声。

三个要钉住的性质：

1. **回路确实闭着**：打开它，她的轨迹必须与关掉时不同（否则只是写日志）。
2. **打分不看"她还在不在原地"**：实测（2026-11-12）按终点打分时 22 次预演全中、准确率 1.0，
   那不是预演准，是"她会待着"这件事本身准。现在只在她（或预演）确实离开时才计分，
   "两边都没走"记为 trivial，不计入准确率。
3. **有界**：预演的增益也是 [0.8, 1.4]，所以探索倍率始终落在 [0.64, 1.96]，不会失控。

Run: .\.venv\Scripts\python.exe tests\test_loop.py
"""

from __future__ import annotations

import collections
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

from src.L5_cognition.stream import Stream  # noqa: E402


def _warm(steps: int = 200, amplitude: float = 0.02) -> Stream:
    from src.common import persist

    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive for this test")
    for _ in range(steps):
        st.landscape.set_noise_amplitude(amplitude)
        st.step()
    return st


# ══════════════════════════════════════════════════════════════════════════════
# 1. 预演多远由她自己的停留决定
# ══════════════════════════════════════════════════════════════════════════════

def test_horizon_comes_from_her_own_dwelling() -> None:
    st = _warm(40)
    st._dwell_history = collections.deque([3, 3, 9], maxlen=64)
    assert st._rehearse_horizon() == 3, st._rehearse_horizon()
    st._dwell_history = collections.deque(maxlen=64)
    assert st._rehearse_horizon() == 1, "no history -> one step, not a chosen constant"
    print("       horizon: median of [3,3,9] -> 3; empty -> 1")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 打分对象是"何时离开"
# ══════════════════════════════════════════════════════════════════════════════

def test_exit_detection_ignores_staying() -> None:
    f = Stream._first_exit
    assert f([5, 5, 5], 5) is None, "never left"
    assert f([5, 5, 7], 5) == 3, f([5, 5, 7], 5)
    assert f([9], 5) == 1
    print("       _first_exit: stay -> None, leave at 3rd -> 3")


def test_trivial_rehearsals_are_not_scored() -> None:
    st = _warm(40)
    st.rehearse_every = 8
    for _ in range(240):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    scored = st.rehearsal_total
    trivial = st.rehearsal_trivial
    assert scored + trivial > 0, "the loop never ran"
    assert st.rehearsal_hits <= scored
    print(f"       scored={scored} hits={st.rehearsal_hits} trivial={trivial} "
          f"acc={st.rehearsal_accuracy:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 有界 + 确实改变她
# ══════════════════════════════════════════════════════════════════════════════

def test_exploration_gain_stays_bounded() -> None:
    st = _warm(60)
    for _ in range(120):
        st.landscape.set_noise_amplitude(0.02 * st.exploration_gain)
        st.step()
        g = st.exploration_gain
        assert 0.64 - 1e-9 <= g <= 1.96 + 1e-9, g
    print(f"       exploration_gain within [0.64, 1.96] over 120 steps (now {g:.4f})")


def test_loop_changes_her_trajectory() -> None:
    off, on = _warm(200), _warm(200)
    off.rehearse_every = 0
    on.rehearse_every = 16
    for st in (off, on):
        for _ in range(240):
            st.landscape.set_noise_amplitude(0.02 * st.exploration_gain)
            st.step()
    same = off._st.x.tobytes() == on._st.x.tobytes()
    assert not same, "the loop is not connected: same position after 240 steps"
    assert on.rehearsal_total > 0, "no rehearsal was ever scored"
    print(f"       after 240 steps: off gain={off.exploration_gain:.3f} "
          f"on gain={on.exploration_gain:.3f} (rehearsals={on.rehearsal_total})")


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
