r"""life_audit 的「休息」口径（2026-11-27 修）。

病：工具硬编码 stream.py:232 `_use_rest=False`，早就过期；rest 场 2026-11-20
已经接上（True）。修：动态问 Stream，不看源码行号。

Run: .\.venv\Scripts\python.exe tests\test_life_audit_rest.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.life_audit import _rest_wired  # noqa: E402


def test_stream_rest_is_wired() -> None:
    from src.L5_cognition.stream import Stream
    assert bool(getattr(Stream(), "_use_rest", False)) is True
    print("       Stream._use_rest = True")


def test_audit_helper_matches_stream() -> None:
    assert _rest_wired() is True
    print("       life_audit._rest_wired() = True")


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
