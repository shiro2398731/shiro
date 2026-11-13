r"""Journal replay must reproduce the state exactly.

This is the acceptance test for "continuous": if replay from the journal gives a
different structure than the original run, the journal is a lie and the rollback
window is not really closed.

Run: .\.venv\Scripts\python.exe tests\test_journal_replay.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.common.journal import Journal  # noqa: E402
from src.L5_cognition.terrain import StateSpace  # noqa: E402


def _sig(space: StateSpace) -> tuple[int, float, float]:
    attrs = space.attractors()
    if not attrs:
        return (0, 0.0, 0.0)
    mus = np.stack([a.mu for a in attrs])
    amps = np.array([a.amp for a in attrs])
    return (len(attrs), round(float(mus.sum()), 9), round(float(amps.sum()), 9))


def _run(space: StateSpace, xs: list[np.ndarray], every: int = 16) -> None:
    for i, x in enumerate(xs):
        space.record(x)
        if (i + 1) % every == 0:
            space.recover()


def test_replay_reproduces_state() -> None:
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "l5.wal"
        xs = [rng.normal(size=StateSpace().D) for _ in range(300)]

        live = StateSpace()
        j = Journal(path)
        live.journal = j
        _run(live, xs)
        j.flush()
        j.close()
        live.journal = None
        sig_live = _sig(live)

        replayed = StateSpace()
        replayed.journal = None
        counts = Journal(path).replay(replayed)
        sig_replay = _sig(replayed)

        assert counts["observations"] == 300, counts
        assert counts["recovers"] == 300 // 16, counts
        assert sig_live == sig_replay, f"重放不一致:\n  live   ={sig_live}\n  replay ={sig_replay}"
        print(f"       sig={sig_live}  records={counts}")


def test_torn_tail_is_dropped_not_repaired() -> None:
    rng = np.random.default_rng(11)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "l5.wal"
        xs = [rng.normal(size=StateSpace().D) for _ in range(50)]
        live = StateSpace()
        j = Journal(path)
        live.journal = j
        _run(live, xs)
        j.flush()
        j.close()

        raw = path.read_bytes()
        path.write_bytes(raw[:-100])          # simulate a kill mid-write

        space = StateSpace()
        space.journal = None
        counts = Journal(path).replay(space)

        assert counts["observations"] == 49, counts
        assert len(space.attractors()) > 0
        print(f"       survived torn tail: records={counts}")


def test_empty_journal_is_a_noop() -> None:
    with tempfile.TemporaryDirectory() as d:
        space = StateSpace()
        space.journal = None
        counts = Journal(pathlib.Path(d) / "l5.wal").replay(space)
        assert counts["observations"] == 0 and counts["recovers"] == 0
        assert len(space.attractors()) == 0


def test_one_bad_record_voids_the_whole_log() -> None:
    """Fail closed: a log with one unusable record is not replayed at all.

    2026-11-13, measured live: the journal left over from the 11-08 runaway was
    replayed onto a healthy archive. Filtering the ~1e306 records was not
    enough -- the moderately wrong ones from the same diverged run were
    accepted, and her archive grew to 338 regions where a coherent history had
    307. Replaying the survivors rebuilds a body that never existed, so the
    whole log is refused instead.
    """
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "l5.wal"
        live = StateSpace()
        j = Journal(path)
        live.journal = j
        good = [rng.normal(size=StateSpace().D) for _ in range(30)]
        _run(live, good)
        # one absurd but finite observation, then one non-finite
        j.append_observe(np.full(live.D, 1e300))
        poisoned = np.zeros(live.D)
        poisoned[0] = np.inf
        j.open()
        j._fh.write(b"A")
        j._fh.write(poisoned.astype("<f8").tobytes())
        j.flush()
        j.close()

        space = StateSpace()
        space.journal = None
        counts = Journal(path).replay(space)

        assert counts["aborted"] == 1, counts
        assert counts["observations"] == 0, counts
        assert len(space.attractors()) == 0, len(space.attractors())
        print(f"       log refused as a whole ({counts['rejected']} unusable "
              f"records, nothing applied)")


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