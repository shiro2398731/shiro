r"""Folding: bounded body, replay-identical.

折的不是"不重要"，是"不会再被需要"（assigned < A）且"够旧"（age > k * half_life）。
折叠必须入 journal，否则 replay 会重建出一个带回了已折叠区域的、不同的身体。

Run: .\.venv\Scripts\python.exe tests\test_fold.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

from src.common import journal  # noqa: E402
from src.L5_cognition.terrain import Attractor, StateSpace  # noqa: E402


def _sig(space: StateSpace) -> tuple:
    attrs = space.attractors()
    mus = np.stack([a.mu for a in attrs]) if attrs else np.zeros((0, 1))
    amps = np.array([a.amp for a in attrs]) if attrs else np.zeros(1)
    folded = space._folded
    fmass = round(float(sum(f["amp"] for f in folded)), 9)
    return (len(attrs), round(float(mus.sum()), 9), round(float(amps.sum()), 9),
            len(folded), fmass)


def _space_with_stale(n_stale: int = 60, n_fresh: int = 5) -> StateSpace:
    ss = StateSpace()
    # 2026-11-17：摘要不再进载体，被折的区域写进外置笔记本。测试也要给一个。
    from src.common.external_memory import ExternalMemory
    ss.external_memory = ExternalMemory(pathlib.Path(tempfile.mkdtemp()) / "folded.jsonl")
    ss._total_steps = 100_000
    for i in range(n_stale):
        a = Attractor(mu=np.full(ss.D, 0.01 * i), amp=0.2)
        a._assigned = 1
        a._born_step = 0              # very old
        ss._attractors.append(a)
    for i in range(n_fresh):
        a = Attractor(mu=np.full(ss.D, -0.01 * i), amp=0.5)
        a._assigned = 50
        a._born_step = 99_000          # recent and busy
        ss._attractors.append(a)
    return ss


def test_candidates_are_old_and_rarely_reached() -> None:
    ss = _space_with_stale(40, 5)
    ids = ss.fold_candidates(min_assigned=2, age_factor=1.0)
    assert len(ids) == 40, f"expected the 40 stale regions, got {len(ids)}"
    assert all(i < 40 for i in ids), "busy or recent regions must not be candidates"


def test_fold_collapses_and_summarises() -> None:
    ss = _space_with_stale(40, 5)
    removed = ss.fold(min_assigned=2, age_factor=1.0)
    assert removed == 40
    assert len(ss.attractors()) == 5
    # 2026-11-17：摘要不再进载体（`_folded` 恒为空）。信息去外置笔记本，
    # 而且是**完整**的那份——这正是这个测试原本要保的东西。
    assert ss._folded == [], "summaries must not go back into the carrier"
    recs = list(ss.external_memory.read()) if ss.external_memory else []
    assert len(recs) == 40, f"expected 40 records in the notebook, got {len(recs)}"
    assert all(r.get("mu") is not None and len(r["mu"]) == ss.D for r in recs), \
        "the notebook must keep the full centroid, not a rounded summary"
    print(f"       folded 40 -> {len(ss.attractors())} live regions, "
          f"{len(recs)} full records in the notebook, 0 summaries in the carrier")


def test_fold_is_replayable() -> None:
    """同历史 + 同折叠事件 => 同身体。"""
    rng = np.random.default_rng(21)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "l5.wal"
        live = StateSpace()
        j = journal.Journal(path)
        live.journal = j
        for i in range(120):
            live.record(rng.normal(size=live.D))
            if (i + 1) % 16 == 0:
                live.recover()
        # age_factor=0 让"够旧"恒成立，避免手动改 _total_steps：
        # replay 只重放 record/recover/fold，看不到测试对内部计数器的直接修改，
        # 那样算出来的 jitter 会不同（本次踩过）。
        folded = live.fold(min_assigned=3, age_factor=0.0)
        for i in range(30):
            live.record(rng.normal(size=live.D))
        j.flush()
        j.close()
        live.journal = None
        sig_live = _sig(live)

        replayed = StateSpace()
        replayed.journal = None
        counts = journal.Journal(path).replay(replayed)

        assert counts["folds"] == (1 if folded else 0), counts
        assert sig_live == _sig(replayed), f"折叠后重放不一致:\n  live={sig_live}\n  replay={_sig(replayed)}"
        print(f"       folded={folded}, replay={counts}, sig={sig_live}")


def test_bound_growth_under_periodic_folding() -> None:
    """周期性折叠使 n_attractors 不随步数单调上升。"""
    rng = np.random.default_rng(31)
    ss = StateSpace()
    curve = []
    for block in range(12):
        for _ in range(40):
            ss.record(rng.normal(size=ss.D))
        ss._total_steps += 5_000        # 推进年龄，制造陈旧区域
        ss.fold(min_assigned=2, age_factor=1.0)
        curve.append(len(ss.attractors()))
    assert max(curve[-4:]) < max(curve[:4]) + 40, f"growth not bounded: {curve}"
    print(f"       live regions per block: {curve}")


def _overdue_population() -> StateSpace:
    """10 regions seen early and never again, 30 busy and recent (of 100k steps)."""
    ss = StateSpace(dim=8)
    from src.common.external_memory import ExternalMemory
    ss.external_memory = ExternalMemory(pathlib.Path(tempfile.mkdtemp()) / "folded.jsonl")
    ss._total_steps = 100_000
    for i in range(10):
        a = Attractor(mu=np.full(8, 0.1 * (i + 1)), amp=0.2)
        a._assigned, a._born_step, a._last_seen_step = 1, 0, 500
        ss._attractors.append(a)
    for i in range(30):
        a = Attractor(mu=np.full(8, -0.1 * (i + 1)), amp=0.5)
        a._assigned, a._born_step, a._last_seen_step = 30, 99_000, 99_990
        ss._attractors.append(a)
    return ss


def test_fold_auto_is_silent_without_measurements() -> None:
    """No measurements -> no folding, rather than a threshold chosen by hand."""
    small = StateSpace(dim=8)
    small._total_steps = 1000
    a = Attractor(mu=np.zeros(8), amp=0.2)
    a._assigned, a._last_seen_step = 1, 10
    small._attractors.append(a)
    assert small.fold_auto() == 0, "one region is not a distribution"

    unknown = _overdue_population()
    for attr in unknown.attractors():
        attr._last_seen_step = 0          # legacy record: last visit unknown
    assert unknown.fold_auto() == 0, "unknown last visit must not be read as 'long ago'"
    assert len(unknown.attractors()) == 40
    print("       1 region / unknown last visit: 0 folded")


def test_fold_auto_folds_the_overdue_tail() -> None:
    ss = _overdue_population()
    removed = ss.fold_auto()
    assert removed == 10, f"expected the 10 never-returned regions, got {removed}"
    assert len(ss.attractors()) == 30, len(ss.attractors())
    assert ss._folded == [], "summaries must not go back into the carrier"
    if ss.external_memory is not None:
        assert ss.external_memory.count() == 10
    print(f"       folded {removed} overdue -> {len(ss.attractors())} live regions "
          f"(notebook: {ss.external_memory.count() if ss.external_memory else 0})")


def test_fold_auto_keeps_recently_visited() -> None:
    """She is still coming back: nothing folds, whatever the age is."""
    ss = _overdue_population()
    for i, a in enumerate(ss.attractors()):
        a._assigned = 30
        a._born_step = 0
        a._last_seen_step = 99_999
    assert ss.fold_auto() == 0
    print("       all regions seen at step 99999: 0 folded")


def test_age_criterion_folds_nothing_on_the_same_population() -> None:
    """Why fold_auto uses elapsed-since-visit instead of age since birth.

    On this population the age criterion specified in HANDOFF §3.3 cannot fire:
    the rarely visited regions are their own oldest sample, so p90(interval)
    equals the age it is compared against (measured: cutoff 100000.0, age
    100000 -> 0 candidates).
    """
    ss = _overdue_population()
    now = ss._total_steps
    counts = [a._assigned for a in ss.attractors()]
    assign_ceiling = max(1, int(np.median(counts)))
    intervals = [max(now - a._born_step, 1) / max(a._assigned, 1)
                 for a in ss.attractors()]
    cutoff = float(np.percentile(intervals, 90))
    candidates = [i for i, a in enumerate(ss.attractors())
                  if a._assigned < assign_ceiling and (now - a._born_step) > cutoff]
    assert cutoff == float(now), cutoff
    assert candidates == [], candidates
    print(f"       age criterion: cutoff={cutoff:.0f} candidates=0; "
          f"fold_auto folds 10")


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