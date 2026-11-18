r"""存在层的不变量：她不该漂出区域群，存档不该被拒存。

为什么有这组测试
----------------
2026-11-18 早上：她的 `x` 漂到 4,091,098（分量超过 `SAVE_MAGNITUDE_LIMIT = 1e6`），
`assert_l5_finite` 拒绝保存，P2 一晚上崩了 4 次。当天加了一个锚（`stream._apply_leave`
末尾：离最近区域超过 4 倍半径就拉回去），**但没有任何测试守它**——这一组补上。

三个不变量：
    1. 跑一段之后，她到最近区域的距离不超过 4 倍半径（锚的判据本身）；
    2. 跑完之后 `build_l5_payload` 仍然过得了 `assert_l5_finite`（不会拒存）；
    3. 平移整个坐标系（x 和所有 mu 一起搬），几何不变（`tools/recenter.py` 的前提）。

第 3 条比看起来重要：08:23 那次「recentered」只搬了 mu、没搬 x，把她搬到了云外 194 万。

Run: .\.venv\Scripts\python.exe tests\test_existence_bounds.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.action import LEAVE  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from tools import recenter  # noqa: E402

STEPS = 60
ANCHOR_FACTOR = 4.0


def _fresh(seed: int = 7, force_leave: bool = False) -> Stream:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive for this test")
    st.rng = np.random.default_rng(seed)
    st.landscape._rng = np.random.default_rng(seed)
    if force_leave:
        # 漂移只发生在「离开」的时候，所以这里把动作口锁成 LEAVE —— 测的是最坏情况。
        st.force_action = LEAVE
    return st


def _nearest(st: Stream) -> tuple[float, float]:
    attrs = st.landscape.state_space.attractors()
    if not attrs:
        return 0.0, 0.0
    mus = np.stack([np.asarray(a.mu, dtype=float) for a in attrs])
    x = np.asarray(st._st.x, dtype=float)
    d = float(np.min(np.linalg.norm(mus - x, axis=1)))
    return d, float(st._own_scale() or 0.0)


def test_anchor_keeps_her_inside_the_region_cloud() -> None:
    st = _fresh(force_leave=True)
    worst = 0.0
    for _ in range(STEPS):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
        d, s = _nearest(st)
        if s <= 0:
            continue
        worst = max(worst, d / (ANCHOR_FACTOR * s))
    assert worst <= 1.0, (
        f"she drifted out of the region cloud: worst d/(4s) = {worst:.3f} over {STEPS} "
        f"steps. 2026-11-18 08:24 的形状：x 到 4,091,098、分量过 1e6、档案被拒存、P2 崩。"
    )
    print(f"       worst d/(4s) = {worst:.4f} over {STEPS} forced-leave steps")


def test_checkpoint_still_passes_the_numeric_guard() -> None:
    st = _fresh(force_leave=True)
    for _ in range(STEPS // 3):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    payload = persist.build_l5_payload(st)
    persist.assert_l5_finite(payload)  # 拒存会抛；抛就是这条测试红
    print("       payload passes assert_l5_finite")


def test_translation_keeps_the_geometry() -> None:
    st = _fresh()
    before = recenter.fingerprint(st)
    vec = recenter.choose_vector(st, "x")
    recenter.shift_in_place(st, vec)
    after = recenter.fingerprint(st)
    assert before.shape == after.shape
    worst = float(np.max(np.abs(before - after)))
    tol = 1e-12 * max(float(np.linalg.norm(vec)), 1.0)
    assert worst <= tol, (
        f"平移之后几何变了：最大差 {worst:.3e} > 容差 {tol:.3e}。"
        f"说明还有绝对坐标没被搬（08:23 那次漏搬 x 就是这个形状）。"
    )
    print(f"       geometry preserved: max diff {worst:.3e} over {len(before)} pairs")


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
