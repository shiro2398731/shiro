r"""show.py 走速行的测试（2026-11-28）。

心跳新鲜不等于她在走；这个测试钉住「步数增量/步速/观测年龄」的计算。

Run: .\.venv\Scripts\python.exe tests\test_show_progress.py
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.show import motive_stats, p4_age_min, progress_from  # noqa: E402


def test_p4_age() -> None:
    p = __import__("pathlib").Path("data/state/_p4_age_test.hb")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="utf-8")
    __import__("os").utime(p, (0, __import__("time").time() - 120))
    age = p4_age_min(str(p))
    assert age is not None and 1.8 < age < 2.2, age
    p.unlink()
    assert p4_age_min(str(p)) is None
    print(f"       p4 age ok ({age:.2f} min)")


def test_motive_tiny_margin() -> None:
    hist = [100.0, 100.05, 100.02, 100.08, 100.04]
    g = motive_stats(hist)
    assert g["n"] == 5
    assert g["margin_rel"] < 0.01, g
    assert g["cv"] < 0.01, g
    print(f"       motive margin={g['margin_rel']:.4%} cv={g['cv']:.4%}")


def test_motive_empty() -> None:
    assert motive_stats([]) == {}
    print("       empty history -> {}")


def test_progress_normal() -> None:
    now = datetime(2026, 11, 28, 5, 0, 0, tzinfo=timezone.utc)
    rows = [(now - timedelta(seconds=120), 100),
            (now - timedelta(seconds=60), 130)]
    g = progress_from(rows, now)
    assert g["delta"] == 30, g
    assert g["rate_per_min"] == 30.0, g
    assert g["obs_age_s"] == 60.0, g
    print(f"       delta={g['delta']} rate={g['rate_per_min']}/min")


def test_insufficient_rows() -> None:
    assert progress_from([], None) == {}
    assert progress_from([(datetime.now(timezone.utc), 1)], None) == {}
    print("       <2 rows -> {}")


def test_flat_then_stale() -> None:
    now = datetime(2026, 11, 28, 5, 0, 0, tzinfo=timezone.utc)
    rows = [(now - timedelta(seconds=3600), 100),
            (now - timedelta(seconds=1200), 100)]
    g = progress_from(rows, now)
    assert g["delta"] == 0, g
    assert g["obs_age_s"] == 1200.0, g
    print(f"       delta=0, obs_age={g['obs_age_s']}s -> show 会标停摆")


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
