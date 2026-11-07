r"""P0-2 的验收：出生深度不该随运行时长衰减。

对应评估文档 §3.2 的验收条件：
  * 10^6 步时 birth_depth 不在 min_amp 之下（旧公式会）
  * birth_depth(10^6) / birth_depth(10^4) 落在 [0.5, 2.0]（尺度无关）

运行：.\.venv\Scripts\python.exe tests\test_birth_depth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.L5_cognition.terrain import Attractor, StateSpace  # noqa: E402

# 观测到的一个真实量级：147664 步里某区域被访问 5 次 → 3.4e-5。
# 这里用一个略宽的典型值，避免拿单点当总体。
POP_RATE = 1.0e-4
POP_N = 40


def _space(total_steps: int, rate: float = POP_RATE, n: int = POP_N) -> StateSpace:
    """造一个"已经跑过 total_steps 步、有 n 个区域"的状态空间。"""
    ss = StateSpace()
    ss._total_steps = total_steps
    rng = np.random.default_rng(7)
    for _ in range(n):
        a = Attractor(mu=rng.normal(size=ss.D), amp=0.5)
        a._born_step = 0
        a._assigned = max(int(rate * total_steps), 1)
        ss._attractors.append(a)
    return ss


def _old_birth_depth(ss: StateSpace) -> float:
    """改动前的公式，作对照：p = 1 / total_steps。"""
    p = 1.0 / max(int(ss._total_steps), 1)
    num = p * ss.assign_ratio
    R = ss.recover_rate / max(int(ss._recover_every_hint), 1)
    return float(ss.ceiling * num / (num + R))


def test_old_formula_collapses_with_elapsed_time() -> None:
    d4 = _old_birth_depth(_space(10_000))
    d6 = _old_birth_depth(_space(1_000_000))

    assert d4 > 1e-3, d4
    assert d6 < 1e-3, f"旧公式在 1e6 步时应当低于 min_amp，实际 {d6:.3e}"
    assert d6 < d4 / 50, "旧公式随总步数衰减两个数量级"


def test_new_formula_is_scale_free() -> None:
    new = _space(10_000)
    old = _space(1_000_000)

    d4 = new._birth_depth(1.0)
    d6 = old._birth_depth(1.0)

    assert 0.5 <= (d6 / d4) <= 2.0, f"出生深度依赖运行时长：{d4:.4e} -> {d6:.4e}"


def test_new_regions_survive_at_one_million_steps() -> None:
    ss = _space(1_000_000)

    depth = ss._birth_depth(1.0)

    assert depth >= ss.min_amp, f"{depth:.3e} < min_amp {ss.min_amp:.1e}（出生即死）"


def test_cold_start_falls_back_to_one_assignment_per_step() -> None:
    ss = StateSpace()
    ss._total_steps = 500
    assert ss._population_visit_rate() is None, "区域不足 8 个时不该能测出群体访问率"

    depth = ss._birth_depth(1.0)

    assert depth > ss.min_amp, depth


def main() -> int:
    print("birth_depth 对照表（POP_RATE = %.0e，%d 个区域）\n" % (POP_RATE, POP_N))
    print(f"{'total_steps':>12} {'旧公式 p=1/total':>18} {'新公式 群体实测率':>18} {'min_amp':>9}")
    for total in (10_000, 50_000, 147_664, 400_000, 1_000_000, 10_000_000):
        ss = _space(total)
        print(f"{total:>12,} {_old_birth_depth(ss):>18.3e} {ss._birth_depth(1.0):>18.3e} {ss.min_amp:>9.1e}")
    print()

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