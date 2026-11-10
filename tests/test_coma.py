r"""停摆即昏迷：时间账本的单元测试。

运行方式（项目 venv 里没有 pytest，所以两头都能跑）：

    .\.venv\Scripts\python.exe tests\test_coma.py
    # 装了 pytest 时：pytest tests/test_coma.py

覆盖评估文档 §5.6 的第 1/2/3/5 条验收。
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import coma  # noqa: E402


class _Acc:
    """假 SlowAccumulator：只需要 P、_hist、HIST 三样。"""

    HIST = 200

    def __init__(self, p: float, hist: list[float]) -> None:
        self.P = p
        self._hist = list(hist)


def _write_coma(dirpath: str, **fields) -> None:
    state = coma.ComaState(**fields)
    coma.save(state, dirpath)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 首次运行
# ══════════════════════════════════════════════════════════════════════════════

def test_first_run_has_no_gap() -> None:
    with tempfile.TemporaryDirectory() as d:
        state, gap = coma.begin_session(d, activity_paths=())

        assert gap == 0.0
        assert state.session_id == 1
        assert state.coma_events == 0
        assert state.saved_at is not None


# ══════════════════════════════════════════════════════════════════════════════
# 2. 快启 vs 昏迷（§5.6 第 1、2 条）
# ══════════════════════════════════════════════════════════════════════════════

def test_quick_restart_counts_seconds_but_no_event() -> None:
    with tempfile.TemporaryDirectory() as d:
        past = datetime.now(timezone.utc) - timedelta(seconds=30)
        _write_coma(d, saved_at=past.isoformat(), session_id=0)

        state, gap = coma.begin_session(d, activity_paths=())

        assert 29 < gap < 40, gap
        assert state.coma_events == 0, "30s 的快启不该记成昏迷事件"
        assert abs(state.coma_s - gap) < 1e-6, "但那 30 秒确实不在，要计入 coma_s"


def test_coma_after_one_hour() -> None:
    with tempfile.TemporaryDirectory() as d:
        past = datetime.now(timezone.utc) - timedelta(seconds=3600)
        _write_coma(d, saved_at=past.isoformat(), session_id=7, coma_s=100.0, coma_events=2)

        state, gap = coma.begin_session(d, activity_paths=())

        assert abs(gap - 3600) < 5, gap
        assert state.coma_events == 3
        assert state.session_id == 8
        assert abs(state.coma_s - (100.0 + gap)) < 1e-6


def test_rewound_is_separate_from_gap() -> None:
    """回滚 != 昏迷：落盘之后还在写日志的那段，是被丢弃的清醒时间。"""
    import os

    with tempfile.TemporaryDirectory() as d:
        now = datetime.now(timezone.utc)
        _write_coma(d, saved_at=(now - timedelta(seconds=100)).isoformat(), session_id=0)

        events = Path(d) / "events.jsonl"
        events.write_text("{}\n", encoding="utf-8")
        mtime = (now - timedelta(seconds=10)).timestamp()
        os.utime(events, (mtime, mtime))

        state, gap = coma.begin_session(d, activity_paths=(str(events),))

        assert 8 <= gap <= 15, f"gap 应是'真正没在跑'的那段，实际 {gap}"
        assert 85 <= state.last_rewound_s <= 95, f"回滚量应约 90 s，实际 {state.last_rewound_s}"
        assert abs(state.rewound_s - state.last_rewound_s) < 1e-6


def test_no_rewound_when_logs_are_older_than_checkpoint() -> None:
    import os

    with tempfile.TemporaryDirectory() as d:
        now = datetime.now(timezone.utc)
        _write_coma(d, saved_at=(now - timedelta(seconds=10)).isoformat(), session_id=0)

        events = Path(d) / "events.jsonl"
        events.write_text("{}\n", encoding="utf-8")
        mtime = (now - timedelta(seconds=100)).timestamp()
        os.utime(events, (mtime, mtime))

        state, gap = coma.begin_session(d, activity_paths=(str(events),))

        assert state.last_rewound_s == 0.0, state.last_rewound_s
        assert 8 <= gap <= 15, gap


def test_session_id_increments_across_crashes() -> None:
    with tempfile.TemporaryDirectory() as d:
        ids = []
        for _ in range(3):
            state, _gap = coma.begin_session(d, activity_paths=())
            coma.mark_saved(d)
            ids.append(state.session_id)

        assert ids == [1, 2, 3]


# ══════════════════════════════════════════════════════════════════════════════
# 3. 落盘时间章是幂等的（§5.6 第 5 条的一半：不引入隐藏状态）
# ══════════════════════════════════════════════════════════════════════════════

def test_mark_saved_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as d:
        coma.begin_session(d, activity_paths=())
        before = coma.load(d)

        for _ in range(5):
            coma.mark_saved(d)

        after = coma.load(d)

        assert after.uptime_s == before.uptime_s, "mark_saved 不该累加 uptime"
        assert after.session_id == before.session_id
        assert after.coma_s == before.coma_s


def test_total_uptime_includes_current_session() -> None:
    with tempfile.TemporaryDirectory() as d:
        state, _ = coma.begin_session(d, activity_paths=())
        state.uptime_s = 3600.0
        state.last_boot_at = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        coma.save(state, d)

        total = coma.total_uptime_s(coma.load(d))

        assert 3609 < total < 3620, total


def test_corrupt_ledger_does_not_block_startup() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / coma.COMA_FILE
        p.write_text("{ this is not json", encoding="utf-8")

        state, gap = coma.begin_session(d, activity_paths=())

        assert gap == 0.0
        assert state.session_id == 1
        assert (Path(d) / (coma.COMA_FILE + ".corrupt")).exists(), "坏文件要留作证据"


# ══════════════════════════════════════════════════════════════════════════════
# 4. 零驱动松弛：按被补偿对象自己的 decay 规则算
# ══════════════════════════════════════════════════════════════════════════════

def test_accumulator_settles_to_its_own_history_median() -> None:
    acc = _Acc(p=0.80, hist=[0.20] * 200)

    rep = coma.apply_accumulator_coma(acc, 3600.0)

    assert rep["applied"] is True
    assert rep["reason"] == "settled"
    assert abs(rep["target"] - 0.20) < 1e-12
    assert abs(acc.P - 0.20) < 1e-9, acc.P


def test_accumulator_short_gap_uses_same_decay_rule() -> None:
    acc = _Acc(p=0.80, hist=[0.20] * 200)

    rep = coma.apply_accumulator_coma(acc, 0.32)  # ≈ 200 步 @1.60 ms

    expected = 0.20 + (0.80 - 0.20) * ((1.0 - 1.0 / 200.0) ** 200)
    assert rep["reason"] == "partial"
    assert abs(acc.P - expected) < 1e-9, (acc.P, expected)


def test_accumulator_uses_measured_step_rate_when_available() -> None:
    """有实测步率时不该再用写死的 STEP_PERIOD_S。"""
    acc = _Acc(p=0.80, hist=[0.20] * 200)

    rep = coma.apply_accumulator_coma(acc, 0.40, steps_per_second=500.0)  # 500 步/s → 200 步

    assert rep["steps"] == 200, rep["steps"]
    expected = 0.20 + (0.80 - 0.20) * ((1.0 - 1.0 / 200.0) ** 200)
    assert abs(acc.P - expected) < 1e-9


def test_no_gap_leaves_accumulator_untouched() -> None:
    acc = _Acc(p=0.80, hist=[0.20] * 200)

    rep = coma.apply_accumulator_coma(acc, 0.0)

    assert rep["applied"] is False
    assert acc.P == 0.80


def test_empty_history_is_safe() -> None:
    acc = _Acc(p=0.42, hist=[])

    rep = coma.apply_accumulator_coma(acc, 3600.0)

    assert rep["applied"] is False
    assert acc.P == 0.42


# ══════════════════════════════════════════════════════════════════════════════

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