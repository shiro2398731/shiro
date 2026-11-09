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
    assert len(ss._folded) == 40
    assert abs(sum(f["amp"] for f in ss._folded) - 40 * 0.2) < 1e-9
    print(f"       folded 40 -> {len(ss.attractors())} live regions, "
          f"{len(ss._folded)} summaries")


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