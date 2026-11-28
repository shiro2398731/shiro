r"""L4→L5 审计工具的测试（2026-11-28）。

Run: .\.venv\Scripts\python.exe tests\test_l4_l5_audit.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.l4_l5_audit import Row, cosine, summarize  # noqa: E402


def test_cosine() -> None:
    assert abs(cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) - 1.0) < 1e-9
    assert abs(cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-9
    assert cosine(np.zeros(2), np.array([1.0, 0.0])) == 0.0
    print("       cosine parallel/orthogonal/zero ok")


def _row(t: int, action: int, tension: float, med: float, n: int, s: float) -> Row:
    return Row(t=t, action=action, tension_used=tension, past_median=med, n=n,
               sum=s, med=0, p90=0, mx=0, drive_norm=0.1, disp=1.0,
               cos_drive_disp=0.0, factor_norm=1.0, rest_mean=0.1,
               energy=1.0, uid=1, target_uid=None, cos_drive_target=0.0,
               dsum=0.0)


def test_summarize_flags_tiny_margin() -> None:
    rows = [_row(i, 1 if i % 2 else 0, 100.0 + (i % 2), 100.0, 10, 100.0 + (i % 2))
            for i in range(10)]
    s = summarize(rows)
    assert s["leave_rule_consistency"] == 1.0, s
    assert s["tension_sum_cv"] < 0.01, s
    assert s["margin_rel_median"] < 0.01, s
    print(f"       tiny-margin signal detected (cv={s['tension_sum_cv']})")


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
