r"""daily_check 的测试：五个数都要能出来，而且互不矛盾。

它存在的意义是「每天跑一条命令就知道她有没有犯昨天那种病」。所以这里只钉两件事：
    1. run() 不抛、五个键都在；
    2. 「健康」判定和数值自洽（比如 payload_ok 必须等于 payload_mb < 18）。

Run: .\.venv\Scripts\python.exe tests\test_daily_check.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.daily_check import run  # noqa: E402


def test_five_numbers_are_present() -> None:
    r = run()
    for k in ("visited_hits", "payload_mb", "max_component",
              "action_driven_15min", "regression"):
        assert k in r, k
    assert "regions" in r
    print(f"       visited={r['visited_hits']} payload={r['payload_mb']} MB "
          f"maxcomp={r['max_component']} motion={r['action_driven_15min']}")


def test_health_flags_are_consistent() -> None:
    r = run()
    assert r["payload_ok"] == (r["payload_mb"] < 18.0)
    assert r["max_component_ok"] == (r["max_component"] < 5e5)
    assert r["motion_ok"] == (r["action_driven_15min"] > 0)
    assert r["visited_ok"] == (r["visited_hits"] > 0)
    print(f"       flags consistent: payload={r['payload_ok']} maxcomp={r['max_component_ok']} "
          f"motion={r['motion_ok']} visited={r['visited_ok']}")


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
