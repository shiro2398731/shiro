r"""Developmental stages: her own distribution decides, and the criteria get fed.

Two problems, two groups of tests.

1. The old criteria compared her against numbers the author picked
   (`0.05`, `0.95`, `2`, `0.9`). docs §11 calls that C3: "which stage she is in"
   produced by hand. The AST test below fails if any criterion body goes back to
   containing a float literal.
2. `StageMonitor.observe` was called from P3 with `dev` only, so three of the
   four criteria had empty buffers forever and she could never leave stage 0.
   The behaviour tests feed the four signals the way P3 now does (dev from
   `l2.state.field`, error from `l1.state.surprise`, recurrence and dwell
   computed from `l5.state.stream`).

Run: .\.venv\Scripts\python.exe tests\test_stages_selfderived.py
"""

from __future__ import annotations

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.L7_lifecycle import stages as stages_mod  # noqa: E402
from src.L7_lifecycle.stages import StageMonitor  # noqa: E402

STAGES_PY = pathlib.Path(stages_mod.__file__)

# Criteria plus the evaluation itself: none of these may contain a float
# literal. Ints are allowed (an index, a zero, "at least one").
CRITERIA = ("_dev_not_flat", "_learning_visible", "_has_recurring",
            "_structure_holds", "evaluate")


def _fn(name: str) -> ast.FunctionDef:
    tree = ast.parse(STAGES_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "StageMonitor":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"StageMonitor.{name} not found")


def _float_literals(fn: ast.FunctionDef) -> list[float]:
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    out: list[float] = []
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, float):
                out.append(sub.value)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. the criteria contain no chosen numbers
# ══════════════════════════════════════════════════════════════════════════════

def test_criteria_have_no_float_literals() -> None:
    stray: dict[str, list[float]] = {}
    for name in CRITERIA:
        lits = _float_literals(_fn(name))
        if lits:
            stray[name] = lits
    assert not stray, f"chosen numbers still decide her stage: {stray}"
    print(f"       {len(CRITERIA)} criterion bodies: no float literals")


# ══════════════════════════════════════════════════════════════════════════════
# 2. behaviour: the same feeds P3 produces
# ══════════════════════════════════════════════════════════════════════════════

W = stages_mod.WINDOW
H = stages_mod.HIST


def _improving(k: int, i: int) -> dict:
    """One observation of an improving run.

    `dev` alternates inside the window and the amplitude grows window by
    window: the "not flat" criterion measures dispersion *within* a window, so
    a signal that only drifts between windows would keep scoring zero
    dispersion and read as pinned.
    """
    amp = 0.1 * (1 + k)
    return {"dev": 1.0 + amp * (1 if i % 2 else -1),
            "s_avg": 1.0 / (1.0 + k),
            "recurrence": 0.2 + 0.1 * k,
            "dwell": 1.0 + 0.5 * k}


def _feed_windows(mon: StageMonitor, windows: int, shape) -> None:
    for k in range(windows):
        for i in range(W):
            mon.observe(**shape(k, i))


def test_steep_improvement_reaches_the_top_stage() -> None:
    mon = StageMonitor()
    windows = H // W + 2
    _feed_windows(mon, windows, _improving)
    idx = mon.evaluate("t")
    assert idx == 4, (idx, mon.last_reasons)
    print(f"       improving: stage {idx} ({mon.stage_def().name}), "
          f"criteria {mon.last_reasons}")


def test_flat_signals_stay_at_stage_zero() -> None:
    mon = StageMonitor()
    _feed_windows(mon, H // W + 2,
                  lambda k, i: {"dev": 1.0, "s_avg": 1.0,
                                "recurrence": 0.0, "dwell": 1.0})
    idx = mon.evaluate("t")
    assert idx == 0, (idx, mon.last_reasons)
    print(f"       constant signals: stage {idx}, criteria {mon.last_reasons}")


def test_one_signal_is_not_enough() -> None:
    """Only dev is fed, which is exactly what P3 used to do."""
    mon = StageMonitor()
    _feed_windows(mon, H // W + 2, lambda k, i: {"dev": 1.0 + 0.2 * k})
    idx = mon.evaluate("t")
    assert idx == 0, (idx, mon.last_reasons)
    print(f"       dev only: stage {idx} (as it was before the wiring fix)")


def test_stage_never_goes_backwards() -> None:
    mon = StageMonitor()
    _feed_windows(mon, H // W + 2, _improving)
    high = mon.evaluate("t1")
    assert high > 0
    _feed_windows(mon, 4,
                  lambda k, i: {"dev": 1.0, "s_avg": 10.0,
                                "recurrence": 0.0, "dwell": 1.0})
    assert mon.evaluate("t2") == high, (high, mon.evaluate("t2"))
    print(f"       stage held at {high} after the signals degraded")


def test_short_history_cannot_advance() -> None:
    """Fewer windows than MIN_WINDOWS: no distribution, no rank, no stage."""
    mon = StageMonitor()
    for _ in range(W * (stages_mod.MIN_WINDOWS - 1)):
        mon.observe(dev=1.0, s_avg=1.0, recurrence=0.0, dwell=1.0)
    assert mon.evaluate("t") == 0


def test_evaluation_is_idempotent_and_deterministic() -> None:
    a, b = StageMonitor(), StageMonitor()
    for mon in (a, b):
        _feed_windows(mon, H // W + 2, _improving)
    assert a.evaluate("t") == b.evaluate("t")
    once = a.evaluate("t")
    assert a.evaluate("t") == once, "second evaluation changed the stage"
    print(f"       deterministic: {once}, and stable on re-evaluation")


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
