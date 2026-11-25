r"""回家规则（A1 候选④）：目标持久化 + 4σ 回家。

2026-11-25 白默选 A。钉住四件事：
1. 第一次 LEAVE 会设下「家」，并选一个目标（不是每步重选）；
2. 目标持久化：同一个目标会被连续走向，直到进入它的半径；
3. 一次外出的最远距离不超过 4σ + 一步（4σ 是已有的锚定半径，不是新数）；
4. 旧存档（没有 home_x）能正常装载，家从 None 开始。

Run: .\.venv\Scripts\python.exe tests\test_home_rule.py
"""

from __future__ import annotations

import copy
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.action import LEAVE  # noqa: E402
from src.common import persist  # noqa: E402
from src.L5_cognition.action import ActionGate  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 600
_PAYLOAD: dict | None = None


def _payload() -> dict:
    global _PAYLOAD
    if _PAYLOAD is None:
        st = Stream()
        if not persist.load_l5(st):
            raise SystemExit("no L5 archive for this test")
        _PAYLOAD = persist.build_l5_payload(st)
    return _PAYLOAD


def _boot() -> Stream:
    st = Stream()
    persist.apply_l5_payload(st, copy.deepcopy(_payload()))
    # 存档里可能带着她运行中的家/目标；测试要一个确定的起点。
    st._home_x = None
    st._leave_target = None
    return st


def test_leave_target_is_chosen_and_kept() -> None:
    """ActionGate：选中的目标会被记住；下一次带着它走，不会换。"""
    g = ActionGate()
    x = np.zeros(8)
    cand = np.array([[100.0] + [0.0] * 7, [-90.0] + [0.0] * 7])
    places = [11, 22]
    d1 = g.direction(x, np.zeros(8), 10.0, neighbours=cand,
                     visited=set(), places=places, prefer_far=False)
    t = g.last_target
    assert d1 is not None and t in places, (d1, t)
    assert abs(float(np.linalg.norm(d1)) - 10.0) < 1e-9, np.linalg.norm(d1)

    d2 = g.direction(x, np.zeros(8), 10.0, neighbours=cand,
                     visited=set(), places=places, prefer_far=False,
                     preferred_target=t)
    assert g.last_target == t, (g.last_target, t)
    assert np.allclose(d1, d2), (d1, d2)
    print(f"       target {t} kept across calls, step {np.linalg.norm(d1):.1f}")


def test_home_is_set_on_first_leave() -> None:
    st = _boot()
    assert st._home_x is None
    # 2026-11-25 夜：她可能正卡在 HOLD 里，三百步都不离开——测试不能依赖
    # 她当时的心情，用干预口强制一次 LEAVE（只用于测试，不写回）。
    st.force_action = LEAVE
    for _ in range(3):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
        if st._home_x is not None:
            break
    assert st._home_x is not None, "forced leave did not set a home"
    print(f"       home set after first leave; dist={np.linalg.norm(st._home_x - st._st.x):.1f}")


def test_excursion_is_bounded_by_4sigma() -> None:
    """任意一步，离家的距离 ≤ 4σ + 一步（σ 取当步她自己的中位半径）。

    强制 LEAVE 是为了排除她卡在 HOLD 的运气；界用**当步**的 σ，因为她的 σ
    会随结构变化（强制走 300 步时实测会变大），用起点的 σ 去套是错的。
    """
    st = _boot()
    x0 = st._st.x.copy()
    st.force_action = LEAVE
    seen = set()
    far = 0.0
    worst = 0.0
    for _ in range(300):
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        far = max(far, float(np.linalg.norm(st._st.x - x0)))
        uid = st.detector.current_attractor
        if uid is not None:
            seen.add(int(uid))
        scale_now = st._own_scale()
        if st._home_x is not None and scale_now:
            hd = float(np.linalg.norm(st._home_x - st._st.x))
            worst = max(worst, hd / float(scale_now))
            assert hd <= 5.5 * float(scale_now), (hd, scale_now)
    assert len(seen) >= 3, f"only {len(seen)} regions reached"
    print(f"       home_dist/σ worst {worst:.2f} (<=5.5); "
          f"max_far from start {far:.1f}; {len(seen)} regions")


def test_old_payload_without_home_loads() -> None:
    p = copy.deepcopy(_payload())
    p.pop("home_x", None)
    st = Stream()
    persist.apply_l5_payload(st, p)
    assert st._home_x is None
    print("       legacy checkpoint: home_x = None (set on next leave)")


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
