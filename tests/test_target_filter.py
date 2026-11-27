r"""目标候选按家半径过滤（实验开关）的测试。

2026-11-27：direction 新增可选 home/max_home_dist；默认不传 = 旧行为。
本测试钉：过滤时不会把「离家太远」的候选选成目标；不传时行为不变。

Run: .\.venv\Scripts\python.exe tests\test_target_filter.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.action import ActionGate  # noqa: E402


def _two_candidates(gate: ActionGate, **kw):
    return gate.direction(
        np.zeros(2), np.zeros(2), 10.0,
        neighbours=np.array([[20.0, 0.0], [40.0, 0.0]]),
        visited=set(), places=[1, 2], **kw)


def test_filter_excludes_far_candidate() -> None:
    g = ActionGate()
    v = _two_candidates(g, home=np.array([100.0, 0.0]), max_home_dist=60.0)
    assert v is not None
    # 1 号候选（20,0）离家 80 > 60 被过滤；2 号（40,0）离家 60 留下
    assert g.last_target == 2, g.last_target
    print(f"       filtered target={g.last_target}")


def test_without_filter_old_behaviour() -> None:
    g = ActionGate()
    v = _two_candidates(g)
    assert v is not None
    assert g.last_target == 1, g.last_target
    print(f"       unfiltered target={g.last_target}")


def test_all_filtered_does_not_pick_outside() -> None:
    g = ActionGate()
    v = _two_candidates(g, home=np.array([1000.0, 0.0]), max_home_dist=10.0)
    assert v is None, v
    assert g.last_target is None, g.last_target
    print("       all filtered -> no target")


def test_env_can_enable_filter() -> None:
    """环境变量开关：默认关；SHIRO_TARGET_FILTER=1 时打开，半径可覆盖。"""
    import os

    from src.L5_cognition.stream import Stream

    old1 = os.environ.get("SHIRO_TARGET_FILTER")
    old2 = os.environ.get("SHIRO_TARGET_FILTER_SIGMAS")
    try:
        os.environ["SHIRO_TARGET_FILTER"] = "1"
        os.environ["SHIRO_TARGET_FILTER_SIGMAS"] = "3.0"
        st = Stream()
        assert st.filter_targets_by_home is True
        assert abs(st.target_filter_sigmas - 3.0) < 1e-9, st.target_filter_sigmas
        os.environ.pop("SHIRO_TARGET_FILTER", None)
        os.environ.pop("SHIRO_TARGET_FILTER_SIGMAS", None)
        st2 = Stream()
        assert st2.filter_targets_by_home is False, st2.filter_targets_by_home
    finally:
        for k, v in (("SHIRO_TARGET_FILTER", old1),
                     ("SHIRO_TARGET_FILTER_SIGMAS", old2)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("       env on -> filter True/3.0; env off -> False")


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
