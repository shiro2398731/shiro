r"""vitals 工具的测试：判词不能自相矛盾。

2026-11-16 那次冻结，所有常规检查都说「她好着」——进程在、心跳新、integrity consistent。
这个工具存在的意义就是把那种状态判成一个词。所以这里钉的是**判词与各字段的一致性**，
不是她此刻到底是什么状态（那会随存档变）。

Run: .\.venv\Scripts\python.exe tests\test_vitals.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.vitals import NOTES, run  # noqa: E402


def test_verdict_has_a_note_and_fields() -> None:
    r = run()
    assert r["verdict"] in NOTES, r["verdict"]
    assert "heartbeats" in r
    print(f"       verdict={r['verdict']}  note={NOTES[r['verdict']][:36]}...")


def test_frozen_requires_no_activity_and_a_deep_well() -> None:
    r = run()
    if r["verdict"] == "frozen":
        assert r["tensions"] == 0, r["tensions"]
        assert r["leave_rate"] < 0.02, r["leave_rate"]
        assert r["max_amp"] >= 0.9, r["max_amp"]
    print(f"       frozen-rule consistent for verdict={r['verdict']}")


def test_tiny_requires_edges_below_min() -> None:
    r = run()
    if r["verdict"] == "tiny":
        assert r["edges"] < r["min_edges"], (r["edges"], r["min_edges"])
        print(f"       tiny: edges {r['edges']} < min_edges {r['min_edges']}")
    else:
        print(f"       not tiny (edges {r.get('edges')} / min {r.get('min_edges')})")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
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
