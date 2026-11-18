r"""坐标系归一：搬完之后，她的世界必须还是同一个世界。

2026-11-18 白默：「我不觉得应该有边界。」→ 不设界，只保证「坐标多大都能活」。
`src/L5_cognition/reorigin.py` 在 boot 时把「她 + 地形」整体搬回原点。

这里钉三件事：
    1. 搬完几何不变（两两距离指纹）；08:23 的 recentered 漏搬 x，就是这条红了的样子；
    2. 搬完量级确实下来了（max 分量下降，或者报告「本来就超限」）；
    3. 搬完的 payload 仍然过得了 assert_l5_finite。

Run: .\.venv\Scripts\python.exe tests\test_reorigin.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition import reorigin  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from tools.recenter import fingerprint  # noqa: E402


def _fresh() -> Stream:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive for this test")
    return st


def test_shift_preserves_geometry() -> None:
    st = _fresh()
    before = fingerprint(st)
    vec = reorigin.center_vector(st)
    reorigin.shift_origin(st, vec)
    after = fingerprint(st)
    assert before.shape == after.shape
    worst = float(np.max(np.abs(before - after))) if before.size else 0.0
    tol = 1e-12 * max(float(np.linalg.norm(vec)), 1.0)
    assert worst <= tol, (
        f"搬完几何变了：最大差 {worst:.3e} > 容差 {tol:.3e}。"
        f"说明还有绝对坐标没被搬（08:23 那次漏搬 x 就是这个形状）。"
    )
    print(f"       geometry preserved: max diff {worst:.3e} over {len(before)} pairs")


def test_reorigin_lowers_magnitude() -> None:
    st = _fresh()
    before = reorigin.max_component(st)
    if before <= 1.0:
        print(f"       skipped: max|component| already {before:.3f}")
        return
    info = reorigin.maybe_reorigin(st, limit=before * 0.5)  # 强制触发
    assert info["shifted"], info
    assert info["after"] < before, (before, info["after"])
    print(f"       max|component| {info['before']:,.0f} -> {info['after']:,.0f}"
          f"{' (still over: 地形本身比上限大)' if info.get('still_over') else ''}")


def test_reorigin_is_a_noop_when_small() -> None:
    st = _fresh()
    info = reorigin.maybe_reorigin(st, limit=1e12)
    assert not info["shifted"], info
    print(f"       no-op at limit 1e12 (max {info['max_component']:,.0f})")


def test_payload_survives_the_numeric_guard() -> None:
    st = _fresh()
    reorigin.maybe_reorigin(st, limit=1.0)  # 强制搬一次
    persist.assert_l5_finite(persist.build_l5_payload(st))
    print("       payload passes assert_l5_finite after reorigin")


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
