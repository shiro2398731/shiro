r"""「她自己的推演」原型 #0 的测试（2026-11-27）。

默认关；打开后：
- 每步返回 HOLD/LEAVE 之一，并记录候选与分数；
- 不设置外部目标 `_leave_target`（推演模式下没有「目标」这种东西）；
- 分数被稳健裁剪在合理范围（单个量不许统治总分）。

Run: .\.venv\Scripts\python.exe tests\test_deliberation.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def _boot() -> Stream:
    st = Stream()
    assert persist.load_l5(st), "需要存档"
    return st


def test_default_off() -> None:
    st = Stream()
    assert st.use_deliberation is False, st.use_deliberation
    print("       default use_deliberation=False")


def test_choose_is_bounded_and_records() -> None:
    st = _boot()
    st.use_deliberation = True
    assert st._leave_target is None
    acts = []
    for _ in range(10):
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        acts.append(int(st._action))
        rec = st.deliberation.last
        assert rec and "candidates" in rec, rec
        scores = [c["score"] for c in rec["candidates"]]
        assert all(-6.0 <= s <= 6.0 for s in scores), scores
    assert set(acts) <= {0, 1}, acts
    assert st._leave_target is None, st._leave_target
    print(f"       actions={acts}; candidates ok; no external target")


def test_holds_when_no_attractor() -> None:
    st = Stream()  # 空地形
    st.use_deliberation = True
    from src.L5_cognition.action import HOLD
    action, direction, rec = st.deliberation.choose(st)
    assert action == HOLD and direction is None
    assert rec.get("reason") == "no_attr", rec
    print("       empty space -> HOLD (no_attr)")


def test_rollout_mode_runs() -> None:
    st = _boot()
    st.use_deliberation = True
    st.deliberation.value_mode = True
    st.deliberation.rollout_mode = True
    actions = []
    for _ in range(5):
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        actions.append(int(st._action))
    assert set(actions) <= {0, 1}, actions
    assert st.deliberation.last is not None
    print(f"       rollout actions={actions}, last={st.deliberation.last.get('reason')}")


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
