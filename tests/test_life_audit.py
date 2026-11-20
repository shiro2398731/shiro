"""生命层验收（life_audit.py）的测试。不依赖 pytest。

口径：
    主动  = leave 占比 > 0
    稳态  = P3 的 closure_ok（corr < 0.05 算没断）
    休息  = 测不了（rest.py 的 _use_rest = False）

Run: .\.venv\Scripts\python.exe tests\test_life_audit.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import life_audit  # noqa: E402

_ORIG_LOGS = life_audit.LOGS


def _with_logs(rows_events, rows_obs):
    """临时把 life_audit.LOGS 指到一个临时目录，塞进事件/观测。"""
    td = tempfile.TemporaryDirectory()
    d = pathlib.Path(td.name)
    if rows_events is not None:
        with (d / "events.jsonl").open("w", encoding="utf-8") as f:
            for r in rows_events:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if rows_obs is not None:
        with (d / "observations.jsonl").open("w", encoding="utf-8") as f:
            for r in rows_obs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    life_audit.LOGS = d
    return td


def _iso(offset_s: float = 0.0) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - offset_s))


def test_recent_events_counts_own_vs_noise() -> None:
    rows = []
    for i, nd in enumerate([False, True, True, False]):
        rows.append({"t": _iso(i), "event": "transition", "noise_driven": nd})
    rows.append({"t": _iso(0), "event": "state_flat"})
    td = _with_logs(rows, None)
    try:
        ev = life_audit._recent_events(minutes=20)
    finally:
        life_audit.LOGS = _ORIG_LOGS
        td.cleanup()
    assert ev["transitions"] == 4, ev
    assert ev["own"] == 2, ev
    assert ev["noise"] == 2, ev
    assert ev["flats"] == 1, ev
    print("      转移按 noise_driven 分开数")


def test_recent_events_ignores_old() -> None:
    rows = [{"t": _iso(3600), "event": "transition", "noise_driven": True}]
    td = _with_logs(rows, None)
    try:
        ev = life_audit._recent_events(minutes=20)
    finally:
        life_audit.LOGS = _ORIG_LOGS
        td.cleanup()
    assert ev["transitions"] == 0, ev
    print("      窗口外的旧事件不算")


def test_closure_uses_lifecycle_not_envelope() -> None:
    rows = [
        {"t": _iso(0), "process": "P3", "data": {"phase": "x"}},  # envelope，无 closure
        {"t": _iso(0), "process": "P3", "data": {"closure_ok": True, "corr": -0.3}},
        {"t": _iso(0), "process": "P3", "data": {"closure_ok": True, "corr": -0.1}},
        {"t": _iso(0), "process": "P3", "data": {"closure_ok": False, "corr": 0.8}},
    ]
    td = _with_logs(None, rows)
    try:
        cl = life_audit._closure(hours=24)
    finally:
        life_audit.LOGS = _ORIG_LOGS
        td.cleanup()
    assert cl["n"] == 3, cl
    assert cl["closure_ok"] is True, cl
    print("      稳态只认 lifecycle 快照")


def test_closure_absent_is_none() -> None:
    rows = [{"t": _iso(0), "process": "P3", "data": {"phase": "x"}}]
    td = _with_logs(None, rows)
    try:
        cl = life_audit._closure(hours=24)
    finally:
        life_audit.LOGS = _ORIG_LOGS
        td.cleanup()
    assert cl["closure_ok"] is None, cl
    print("      没有 lifecycle -> None（不假装 OK）")


def test_main_runs() -> None:
    rc = life_audit.main()
    assert rc == 0, rc
    print("      main() 返回 0")


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
