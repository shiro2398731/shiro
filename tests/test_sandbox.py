r"""沙盒：处理历史不能篡改历史。

三条要钉住的东西：

1. **不写回**：`project()` 在副本上跑，跑完之后原件必须逐位不变。记忆若会改动被记的东西，
   那就不是记忆，是篡改。
2. **确定性**：同一具身体、同一段推演，两次结果必须一样（否则"预演"不可复现，也就不可证伪）。
3. **动作真的接在里面**：do(hold) 与 do(leave) 从同一刻出发推同样的步数，落点必须不同 ——
   否则这只手是装饰。

`tools/rehearse.py` 用这些性质去量"预演到底值多少"（实测：14 步内与"原地不动"一样准，
第 15 步起归零，因为离开的时刻是噪声决定的）。
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

from src.L5_cognition.action import HOLD, LEAVE  # noqa: E402
from src.L5_cognition.sandbox import advance, clone, project  # noqa: E402


def _loaded(warm: int = 60):
    from src.common import persist
    from src.L5_cognition.stream import Stream

    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive for this test")
    for _ in range(warm):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    return st


def _fingerprint(st) -> tuple:
    """Everything the projection must not change."""
    return (
        int(st._step_count),
        int(st.landscape.state_space._total_steps),
        len(st.landscape.state_space.attractors()),
        st._st.x.tobytes(),
        st._st.v.tobytes(),
    )


def test_projection_leaves_her_untouched() -> None:
    st = _loaded()
    before = _fingerprint(st)
    project(st, 24, 0.02)
    assert _fingerprint(st) == before, "the sandbox wrote back into her"
    print("       project(24 steps): her state is bit-identical afterwards")


def test_projection_is_deterministic() -> None:
    st = _loaded()
    a = project(st, 32, 0.02)["regions"]
    b = project(st, 32, 0.02)["regions"]
    assert a == b, (a[:8], b[:8])
    print(f"       two projections agree: {a[:6]} ...")


def test_forced_actions_land_differently() -> None:
    st = _loaded()
    hold = project(st, 32, 0.02, force_action=HOLD)["regions"]
    leave = project(st, 32, 0.02, force_action=LEAVE)["regions"]
    assert hold != leave, "the action made no difference in the sandbox"
    print(f"       do(hold) final={hold[-1]}  do(leave) final={leave[-1]}  "
          f"(diverged at step {next(i for i, (x, y) in enumerate(zip(hold, leave)) if x != y) + 1})")


def test_clone_is_a_copy_not_an_alias() -> None:
    st = _loaded()
    c = clone(st)
    before = _fingerprint(st)
    c.landscape.set_noise_amplitude(0.02)
    c.step()
    assert _fingerprint(st) == before, "stepping the clone moved her"
    print("       clone advanced one step; she did not move")


def test_advance_is_not_a_projection() -> None:
    """`advance` really moves her (it is the control), so it must not be used as a sandbox."""
    st = _loaded()
    before = _fingerprint(st)
    advance(st, 5, 0.02)
    after = _fingerprint(st)
    assert after[0] == before[0] + 5, (before[0], after[0])
    assert after[3] != before[3], "her position did not change; advance is not advancing"


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
