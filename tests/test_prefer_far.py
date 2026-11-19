r"""卡住脱困的测试：prefer_far=True 时真的挑「最远可达」，而不是最近。

2026-11-19 21:30 加的机制：连续几次到达都在少数几个地方时，挑最远可达的目标。
它没有自己的测试——只有 test_action 里「不破坏旧语义」那几条（默认 False）。

Run: .\.venv\Scripts\python.exe tests\test_prefer_far.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.action import ActionGate  # noqa: E402


def _scene():
    """她站在原点；三个候选在 1/2/3 个半径外；都没去过。"""
    x = np.zeros(3)
    scale = 1.0
    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    dists = [1.0, 2.0, 3.0]
    neighbours = np.stack([d * u for d, u in zip(dists, dirs)])
    places = ["a", "b", "c"]
    visited = set()
    g = ActionGate()
    return g, x, scale, neighbours, visited, places, dirs, dists


def test_default_picks_nearest() -> None:
    g, x, s, nb, vis, pl, dirs, dists = _scene()
    d = g.direction(x, x, s, neighbours=nb, visited=vis, places=pl)
    assert d is not None
    # 最近的是 a（方向 [1,0,0]）
    assert abs(float(d[0]) - 1.0) < 1e-9, d
    assert abs(float(np.linalg.norm(d)) - s) < 1e-9, d
    print("       default: 朝最近（a）")


def test_prefer_far_picks_farthest() -> None:
    g, x, s, nb, vis, pl, dirs, dists = _scene()
    d = g.direction(x, x, s, neighbours=nb, visited=vis, places=pl, prefer_far=True)
    assert d is not None
    # 最远的是 c（方向 [0,0,1]）
    assert abs(float(d[2]) - 1.0) < 1e-9, d
    assert abs(float(d[0])) < 1e-9, d
    assert abs(float(np.linalg.norm(d)) - s) < 1e-9, d
    print("       prefer_far: 朝最远（c）")


def test_prefer_far_skips_visited() -> None:
    g, x, s, nb, vis, pl, dirs, dists = _scene()
    # c 已经去过 -> 最远的「没去过」变成 b
    vis = {"c"}
    d = g.direction(x, x, s, neighbours=nb, visited=vis, places=pl, prefer_far=True)
    assert d is not None
    assert abs(float(d[1]) - 1.0) < 1e-9, d
    print("       prefer_far + visited: 跳过 c，朝 b")


def test_both_lengths_equal_one_radius() -> None:
    g, x, s, nb, vis, pl, dirs, dists = _scene()
    for pf in (False, True):
        d = g.direction(x, x, s, neighbours=nb, visited=vis, places=pl, prefer_far=pf)
        assert abs(float(np.linalg.norm(d)) - s) < 1e-9, (pf, d)
    print("       both: 步长都等于一个半径（不因 prefer_far 变大）")


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
