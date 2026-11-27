r"""existence_audit 的 visited_places 口径（2026-11-27 修）。

病：2026-11-17 之后 `_visited_places` 存的是区域 uid（int），审计工具还拿质心
tuple 去对 → 恒为 0（真机 628 个去过的地方，对上 0）。修完真机是 618/628。

Run: .\.venv\Scripts\python.exe tests\test_existence_audit_visited.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.existence_audit import _visited_hits  # noqa: E402


class _Attr:
    def __init__(self, uid, mu):
        self.uid = uid
        self.mu = mu


class _Space:
    def __init__(self, attrs):
        self._attrs = attrs

    def attractors(self):
        return self._attrs


class _Landscape:
    def __init__(self, space):
        self.state_space = space


class _Stream:
    def __init__(self, visited, attrs):
        self._visited_places = visited
        self.landscape = _Landscape(_Space(attrs))


MUS = np.array([[0.0, 0.0], [1.0, 1.0]])
ATTRS = [_Attr(1, MUS[0]), _Attr(2, MUS[1])]


def test_uid_visits_match_regions() -> None:
    st = _Stream({1, 2}, ATTRS)
    assert _visited_hits(st, MUS) == 2
    print("       uid visits: 2/2 matched")


def test_stale_tuple_visit_still_matches_by_position() -> None:
    st = _Stream({(0.0, 0.0)}, ATTRS)
    assert _visited_hits(st, MUS) == 1
    print("       old tuple visit matched by position")


def test_unknown_uid_does_not_match() -> None:
    st = _Stream({999}, ATTRS)
    assert _visited_hits(st, MUS) == 0
    print("       unknown uid -> 0")


def test_empty_visits() -> None:
    st = _Stream(set(), ATTRS)
    assert _visited_hits(st, MUS) == 0
    print("       empty visits -> 0")


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
