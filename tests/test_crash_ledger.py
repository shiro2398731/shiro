r"""crash_ledger 的测试：账本要能重算，而且和「崩溃行 + 放弃重启」对得上。

它钉的是**口径**，不是具体数字（日志会长大）：
    1. build() 不抛、键齐全；
    2. 按段分类的异常数 ≤ 崩溃行 + 放弃重启（写坏的 Traceback 头会让它少几条）；
    3. 会话结束数 ≥ 0，且 stopped ≥ fenced。

Run: .\.venv\Scripts\python.exe tests\test_crash_ledger.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.crash_ledger import build  # noqa: E402


def test_ledger_is_self_consistent() -> None:
    r = build()
    if "error" in r:
        print(f"       skipped: {r['error']}")
        return
    for k in ("exceptions", "crash_lines", "gave_up", "sessions_stopped",
              "sessions_fenced", "by_class", "has_dates"):
        assert k in r, k
    total = r["crash_lines"] + r["gave_up"]
    assert r["exceptions"] <= total, (r["exceptions"], total)
    print(f"       exceptions={r['exceptions']} <= crash_lines+gave_up={total}  "
          f"classes={len(r['by_class'])}")


def test_sessions() -> None:
    r = build()
    if "error" in r:
        print("       skipped")
        return
    assert r["sessions_stopped"] >= r["sessions_fenced"] >= 0
    print(f"       sessions: stopped={r['sessions_stopped']} fenced={r['sessions_fenced']}")


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
