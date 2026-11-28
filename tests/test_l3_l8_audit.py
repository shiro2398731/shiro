r"""L3/L8 审计工具的测试（2026-11-28）。

Run: .\.venv\Scripts\python.exe tests\test_l3_l8_audit.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.l3_l5_audit import jaccard as j3, path_of  # noqa: E402
from tools.l8_identity_audit import jaccard as j8  # noqa: E402


def test_jaccard() -> None:
    assert j3(set(), set()) == 1.0
    assert j3({1, 2}, {2, 3}) == 1 / 3
    assert j8({1}, {1}) == 1.0
    print("       jaccard ok")


def test_path_of_handles_none() -> None:
    assert path_of(None) == []

    class R:
        path = (3, 4)

    assert path_of(R()) == [3, 4]
    print("       path_of ok")


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
