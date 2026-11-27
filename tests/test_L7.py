r"""L7 真测试（2026-11-27）：生命周期不是空壳。

原来的 test_L7.py 是 0 字节（src/L7_lifecycle 里 fatigue/sleep 也是 0 字节）。
这里钉：PhaseWitness 记相位/切换率、StageMonitor 评估与往返。

Run: .\.venv\Scripts\python.exe tests\test_L7.py
"""

from __future__ import annotations

import datetime as dt
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L7_lifecycle.rhythm import PhaseWitness  # noqa: E402
from src.L7_lifecycle.stages import StageMonitor  # noqa: E402


def test_phase_witness_tracks_phase() -> None:
    t0 = dt.datetime(2026, 11, 27, 8, 0, 0, tzinfo=dt.timezone.utc)
    w = PhaseWitness()
    for i in range(6):
        w.see("up", 1.0, (t0 + dt.timedelta(seconds=i)).isoformat())
    for i in range(6):
        w.see("down", 1.2, (t0 + dt.timedelta(seconds=6 + i)).isoformat())
    assert w.current_phase == "down", w.current_phase
    med = w.median_period()
    assert med is not None and abs(med - 1.2) < 0.2, med
    rate = w.switching_rate()
    assert rate is not None and 0.0 < rate <= 1.0, rate
    print(f"       phase={w.current_phase}, median={med:.2f}s, switching={rate:.2f}")


def test_stage_monitor_evaluates_and_roundtrips() -> None:
    m = StageMonitor()
    stage = m.evaluate("2026-11-27T08:00:00+00:00")
    assert isinstance(stage, int) and stage >= 0, stage
    d = m.to_dict()
    m2 = StageMonitor.from_dict(d)
    assert m2.current == m.current, (m2.current, m.current)
    sd = m.stage_def()
    assert sd.index == m.current and sd.name, (sd, m.current)
    print(f"       stage={m.current} ({sd.name}), roundtrip ok")


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
