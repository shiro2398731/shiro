r"""merge_dupes.py 的测试：找出来的对必须真的「几乎重合」。

Run: .\.venv\Scripts\python.exe tests\test_merge_dupes.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from tools.merge_dupes import find_pairs  # noqa: E402


def _load():
    st = Stream()
    if not persist.load_l5(st):
        return None
    return st


def test_pairs_are_actually_close() -> None:
    st = _load()
    if st is None:
        print("       skipped: no archive")
        return
    pairs = find_pairs(st, 0.5)
    attrs = st.landscape.state_space.attractors()
    for i, j, d in pairs:
        lim = 0.5 * min(float(attrs[i].sigma), float(attrs[j].sigma))
        assert d < lim, (i, j, d, lim)
        got = float(np.linalg.norm(np.asarray(attrs[i].mu) - np.asarray(attrs[j].mu)))
        assert abs(got - d) < 1e-6, (got, d)
    print(f"       {len(pairs)} pairs, all < 0.5*min(sigma), distances match")


def test_no_self_pairs() -> None:
    st = _load()
    if st is None:
        print("       skipped: no archive")
        return
    pairs = find_pairs(st, 0.5)
    for i, j, _ in pairs:
        assert i != j
    print(f"       no self pairs in {len(pairs)}")


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
