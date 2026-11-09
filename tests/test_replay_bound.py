r"""Invariances that keep replay from degenerating into "reading a context".

1. After a checkpoint the journal is truncated, so a boot replays only what
   happened since that checkpoint. The replay size is bounded by the checkpoint
   interval and does not grow with her age. If this ever fails, she has become
   a system that re-reads her whole life to act once -- i.e. an LLM.

2. A rolling standby is produced from checkpoint files only, and rotation keeps
   exactly N of them.

Run: .\.venv\Scripts\python.exe tests\test_replay_bound.py
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

from src.common import bundle, journal, persist, rolling  # noqa: E402
from src.L5_cognition.terrain import StateSpace  # noqa: E402


def test_replay_size_is_bounded_by_the_last_checkpoint() -> None:
    """200 records, checkpoint (truncate), 50 more: a boot replays 50, not 250."""
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "l5.wal"
        space = StateSpace()
        j = journal.Journal(path)
        space.journal = j
        for _ in range(200):
            space.record(rng.normal(size=space.D))
        j.flush()

        # checkpoint boundary: the state is durable now, the history is not needed
        j.truncate()
        assert j.read_entries()[0] == [] or len(j.read_entries()[0]) == 0

        for _ in range(50):
            space.record(rng.normal(size=space.D))
        j.flush()

        j.close()   # Windows: an open handle blocks TemporaryDirectory cleanup
        obs, rec = j.read_entries()
        assert len(obs) == 50, f"replay would need {len(obs)} records, expected 50"
        assert len(obs) <= 50, "replay size must not carry history from before the checkpoint"
        print(f"       replay window = {len(obs)} records (checkpoint interval bound)")


def test_rotation_keeps_exactly_n() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "bundles"
        for _ in range(5):
            rolling.rotate(out, keep=3, journal_path=None)
        kept = sorted(out.glob("her-*.bundle"))
        assert len(kept) == 3, f"expected 3 standbys, got {len(kept)}"
        assert all(bundle.doctor(p) == [] for p in kept), "a standby failed doctor"
        print(f"       kept {len(kept)}: {[p.name for p in kept]}")


def test_standby_is_loadable_and_complete() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d) / "bundles"
        info = rolling.rotate(out, keep=1, journal_path=None)
        p = pathlib.Path(info["path"])
        meta = bundle.info(p)
        assert meta["doctor"] == "ok", meta["doctor"]
        assert set(meta["layers"]) == {"l0", "l1", "l2", "acc", "l5"}
        print(f"       {p.name}: {meta['attractors']} attractors, "
              f"l0_steps={meta['l0_step_count']}, {info['size_kb']} KB")


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