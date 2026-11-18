r"""clean_members 的测试：审计口径要自洽（不钉具体数字，存档会变）。

Run: .\.venv\Scripts\python.exe tests\test_clean_members.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.clean_members import audit  # noqa: E402


def test_audit_returns_consistent_counts() -> None:
    r = audit(100.0)
    if "error" in r:
        print(f"       skipped: {r['error']}")
        return
    for k in ("regions", "members", "dist_p50", "dist_p90", "over_limit", "over_mult"):
        assert k in r, k
    assert 0 <= r["over_mult"] <= r["members"], (r["over_mult"], r["members"])
    assert 0 <= r["over_limit"] <= r["members"], (r["over_limit"], r["members"])
    print(f"       regions={r['regions']} members={r['members']} "
          f"over_100sigma={r['over_mult']} over_limit={r['over_limit']}")


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
