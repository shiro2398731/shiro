r"""M4：注意瓶颈（容量受限的选择）。

workspace 的判据在 docs §9 里写作「选择熵 / 切换率在区间内」。没有两个端点，
这句话就是空的。这里钉住的是它的四条性质：

1. **点火要超过她自己以往的水平**（参考线是历史峰值的中位数）：强度不变时不会反复点火；
2. **持续靠动力学维持**：她离开一个特征尺度之后，活跃状态就结束（不是一个计时器）；
3. **历史有界**：`history` 是缓冲区（`HISTORY_CAP`），不是无界列表；
4. **容量是 1**：同一时刻只有一个表征被全局广播 —— 这是实现的选择，写成断言免得被误读成"测出来的"。

真正的分布统计在 `tools/m4_audit.py` 里（需要一段真实运行），这里只钉机制。

Run: .\.venv\Scripts\python.exe tests\test_m4.py
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

from src.L5_cognition.terrain import Attractor, StateSpace  # noqa: E402
from src.L5_cognition.workspace import HISTORY_CAP, GlobalWorkspace  # noqa: E402


def _space(n: int = 4, dim: int = 8) -> StateSpace:
    ss = StateSpace(dim=dim)
    for i in range(n):
        a = Attractor(mu=np.full(dim, 2.0 * i), amp=0.5 + 0.1 * i, scale=0.5)
        a.members.extend([np.full(dim, 2.0 * i)] * 8)   # enough for a stable sigma
        ss._attractors.append(a)
    return ss


def test_ignition_must_exceed_her_own_history() -> None:
    ss = _space()
    ws = GlobalWorkspace()
    x = ss.attractors()[3].mu.copy()
    first = ws.observe(ss, x, "t0")
    assert first is None, "with no history there is no reference level to exceed"
    again = [ws.observe(ss, x, f"t{i}") for i in range(1, 6)]
    assert all(g is None for g in again), "a constant state must not keep igniting"
    print(f"       5 observations at a fixed point -> {sum(g is not None for g in again)} ignitions")
    assert ws.ignition_count == 0


def test_ignition_happens_when_she_moves_to_something_stronger() -> None:
    ss = _space()
    ws = GlobalWorkspace()
    weak = ss.attractors()[0].mu.copy()
    for i in range(6):
        ws.observe(ss, weak, f"w{i}")
    strong = ss.attractors()[3].mu.copy()
    got = ws.observe(ss, strong, "s")
    assert got is not None, "a stronger representation should ignite"
    assert got.attractor_index == 3, got.attractor_index
    print(f"       moved to region 3 -> ignited strength {got.strength:.3f}")


def test_sustaining_ends_when_she_leaves_the_scale() -> None:
    ss = _space()
    ws = GlobalWorkspace()
    weak = ss.attractors()[0].mu.copy()
    for i in range(6):
        ws.observe(ss, weak, f"w{i}")
    strong = ss.attractors()[3].mu.copy()
    assert ws.observe(ss, strong, "s") is not None
    assert ws.active, "right after ignition the representation is active"
    far = strong + 10.0
    assert ws.observe(ss, far, "out") is None
    assert not ws.active, "leaving one scale ends the sustain, with no timer involved"
    print("       sustained while inside one sigma; ended on leaving")


def test_capacity_is_one_and_history_is_bounded() -> None:
    ss = _space()
    ws = GlobalWorkspace()
    weak, strong = ss.attractors()[0], ss.attractors()[3]

    # Mostly weak with occasional strong impulses: the bar sits at the 90th
    # percentile of her own peaks, so a strong impulse clears it, while a
    # two-level alternation (50/50) does not -- that is the point of the
    # quantile, and it is why this test cannot just alternate.
    fired = 0
    cycles = HISTORY_CAP + 200
    for i in range(cycles):
        for k in range(10):
            ws.observe(ss, weak.mu.copy(), f"w{i}-{k}")
        got = ws.observe(ss, strong.mu.copy(), f"s{i}")
        if got is not None:
            fired += 1
            assert ws.active and ws.current is not None
        ws._active_mu = None          # pretend she left, so the next one can ignite
        ws._active_sigma = 0.0

    assert fired > HISTORY_CAP, fired
    assert len(ws.history) == HISTORY_CAP, len(ws.history)
    assert ws.history.maxlen == HISTORY_CAP
    print(f"       {fired} ignitions over {cycles} cycles; history capped at "
          f"{HISTORY_CAP}; one active representation at a time")


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
