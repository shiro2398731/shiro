r"""show.py 的测试：一页观察必须是「只读 + 字段齐全」。

Run: .\.venv\Scripts\python.exe tests\test_show.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.show import _one  # noqa: E402


def test_one_returns_observation_fields() -> None:
    r = _one()
    if "error" in r:
        print(f"       skipped: {r['error']}")
        return
    for k in ("regions", "payload_mb", "norm_x", "d_nearest", "leave_share", "steps"):
        assert k in r, k
    assert r["regions"] > 0
    assert 0 <= r["leave_share"] <= 1
    assert r["payload_mb"] > 0
    print(f"       regions={r['regions']} payload={r['payload_mb']}MB "
          f"|x|={r['norm_x']:.0f} leave={r['leave_share']:.3f}")


def test_one_is_read_only() -> None:
    """连读两次，存档的 mtime 不该变（这个函数只读）。"""
    import os
    import time

    from src.common import persist

    path = persist.STATE_DIR / "l5.pkl"
    if not path.exists():
        print("       skipped: no archive")
        return
    before = os.stat(path).st_mtime
    time.sleep(0.05)
    _one()
    after = os.stat(path).st_mtime
    assert before == after, "show._one() 改了存档"
    print("       read-only: archive mtime unchanged")


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
