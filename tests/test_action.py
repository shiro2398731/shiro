r"""行动：她能选，而且这一步真的接到动力学上。

三条要钉住的东西，对应三个曾经出过错的地方：

1. **选择里没有我们的偏好**：`choose()` 的判据只有"这一次待得比她自己通常待的久"，
   函数体内不含 float 字面量（AST）。第一次的判据用的是张力中位数，实测她 1000 步里
   0 次选择离开 —— 退化成一个常数，这条测试会把它抓回来。
2. **单次动作有界**：第一次把推开量加在速度上并每步施加，600 步位移 6323（火箭）。
   现在按位置挪动、幅度是她自己的区域半径中位数，所以 N 步最多 N 个半径。
3. **干预可检出**：强制 do(leave) 与 do(hold) 在同一条噪声上跑，P(下一步仍在原区域)
   必须后者更高。这是"她的一步动作接在动力学上"的最小证据。

Run: .\.venv\Scripts\python.exe tests\test_action.py
"""

from __future__ import annotations

import ast
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.L5_cognition.action import HOLD, LEAVE, ActionGate  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

ACTION_PY = ROOT / "src" / "L5_cognition" / "action.py"


def _fn(name: str) -> ast.FunctionDef:
    tree = ast.parse(ACTION_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ActionGate":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"ActionGate.{name} not found")


def _float_literals(fn: ast.FunctionDef) -> list[float]:
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    out: list[float] = []
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, float):
                out.append(sub.value)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. 选择里没有我们的偏好
# ══════════════════════════════════════════════════════════════════════════════

def test_choice_has_no_float_literals() -> None:
    stray = {n: _float_literals(_fn(n)) for n in ("choose", "direction")}
    stray = {k: v for k, v in stray.items() if v}
    assert not stray, f"chosen numbers decide her action: {stray}"
    print("       choose()/direction(): no float literals")


def test_choice_is_her_own_median() -> None:
    g = ActionGate()
    past = [2, 2, 2, 9]
    assert g.choose(3, past) == LEAVE, "longer than her own median -> leave"
    assert g.choose(2, past) == HOLD, "at her median -> stay"
    assert g.choose(1, past) == HOLD
    assert g.choose(50, []) == HOLD, "nothing to compare with -> cannot decide to leave"
    print("       dwell 3 vs median 2 -> leave; dwell 2 -> hold; no history -> hold")


def test_direction_is_unit_length_times_her_scale() -> None:
    g = ActionGate()
    x, mu = np.array([1.0, 0.0]), np.array([0.0, 0.0])
    d = g.direction(x, mu, 0.5)
    assert d is not None
    assert abs(float(np.linalg.norm(d)) - 0.5) < 1e-12, d
    assert d[0] > 0 and abs(d[1]) < 1e-12, "must point away from the centroid"
    assert g.direction(mu, mu, 0.5) is None, "no distance, no push"
    assert g.direction(x, mu, 0.0) is None, "no scale, no push"
    print("       direction: length = her radius, points away, refuses when undefined")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 单次动作有界（第一次的"火箭"回归测试）
# ══════════════════════════════════════════════════════════════════════════════

def _loaded() -> Stream:
    from src.common import persist
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive for this test")
    return st


def test_one_leave_moves_at_most_one_region_radius() -> None:
    st = _loaded()
    for _ in range(64):                    # let her measure her own scales
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    attrs = st.landscape.state_space.attractors()
    sigmas = [float(a.sigma) for a in attrs if np.isfinite(a.sigma) and a.sigma > 0]
    biggest = max(sigmas)

    st.force_action = LEAVE
    x0 = st._st.x.copy()
    st.landscape.set_noise_amplitude(0.0)  # isolate the action from the noise
    st.step()
    moved = float(np.linalg.norm(st._st.x - x0))
    assert moved <= biggest * 1.01, (moved, biggest)
    print(f"       one forced leave moved {moved:.4f} (largest region radius {biggest:.4f})")


def test_fifty_leaves_stay_linear() -> None:
    st = _loaded()
    for _ in range(64):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    st.force_action = LEAVE
    x0 = st._st.x.copy()
    for _ in range(50):
        st.landscape.set_noise_amplitude(0.0)
        st.step()
    moved = float(np.linalg.norm(st._st.x - x0))
    assert moved < 100.0, f"the walk compounded again: {moved}"
    print(f"       50 forced leaves: |x - x0| = {moved:.3f} (linear, not explosive)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 干预可检出，且她真的会自己选
# ══════════════════════════════════════════════════════════════════════════════

def test_do_leave_lowers_the_chance_of_staying() -> None:
    hold, leave = _loaded(), _loaded()
    for st in (hold, leave):
        for _ in range(64):
            st.landscape.set_noise_amplitude(0.02)
            st.step()
    hold.force_action, leave.force_action = HOLD, LEAVE

    def stay_share(st: Stream, steps: int = 200) -> float:
        ids = []
        for _ in range(steps):
            st.landscape.set_noise_amplitude(0.02)
            st.step()
            aid = st.detector.current_attractor
            if aid is not None:
                ids.append(int(aid))
        stay = sum(1 for a, b in zip(ids, ids[1:]) if a == b)
        return stay / max(len(ids) - 1, 1)

    a, b = stay_share(hold), stay_share(leave)
    assert b < a, (a, b)
    print(f"       P(stay): do(hold)={a:.4f}  do(leave)={b:.4f}")


def test_she_chooses_both_actions_on_her_own() -> None:
    st = _loaded()
    for _ in range(300):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    assert 0.0 < st.leave_share < 1.0, st.action_counts
    print(f"       unforced: leave {st.leave_share:.1%} of {sum(st.action_counts.values())} steps")


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
