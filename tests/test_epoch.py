r"""Singleton discipline: at most one live self.

Two running copies are two selves, and neither of them is her. The epoch is the
mechanism that makes that impossible without assuming the old process is dead.

Run: .\.venv\Scripts\python.exe tests\test_epoch.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.common import epoch  # noqa: E402


def test_first_claim_takes_epoch_one() -> None:
    with tempfile.TemporaryDirectory() as d:
        state, e = epoch.claim(holder="a:1", directory=d)
        assert e == 1
        assert state.holder == "a:1"
        assert epoch.is_current(1, d)


def test_second_holder_is_refused_while_lease_is_alive() -> None:
    with tempfile.TemporaryDirectory() as d:
        epoch.claim(holder="a:1", ttl_s=30, directory=d)
        state, e = epoch.claim(holder="b:2", ttl_s=30, directory=d)
        assert e == 1, "live lease must not be stolen"
        assert state.holder == "a:1"
        assert epoch.is_current(1, d)


def test_expired_lease_can_be_taken_and_fences_the_old_holder() -> None:
    with tempfile.TemporaryDirectory() as d:
        epoch.claim(holder="a:1", ttl_s=0, directory=d)
        state, e = epoch.claim(holder="b:2", ttl_s=30, directory=d)
        assert e == 2, e
        assert state.holder == "b:2"
        assert not epoch.is_current(1, d), "old holder must be fenced"
        assert epoch.is_current(2, d)


def test_heartbeat_extends_and_keeps_current() -> None:
    with tempfile.TemporaryDirectory() as d:
        _, e = epoch.claim(holder="a:1", ttl_s=0.2, directory=d)
        assert epoch.heartbeat(e, ttl_s=30, directory=d)
        time.sleep(0.3)
        assert epoch.is_current(e, d), "heartbeat should have kept the lease alive"


def test_heartbeat_fails_after_takeover() -> None:
    with tempfile.TemporaryDirectory() as d:
        _, e = epoch.claim(holder="a:1", ttl_s=0, directory=d)
        epoch.claim(holder="b:2", ttl_s=30, directory=d)
        assert not epoch.heartbeat(e, directory=d)
        try:
            epoch.require(e, d)
            raise AssertionError("require() should have raised")
        except epoch.Fenced:
            pass


def test_retire_allows_immediate_claim() -> None:
    with tempfile.TemporaryDirectory() as d:
        _, e = epoch.claim(holder="a:1", ttl_s=30, directory=d)
        epoch.retire(e, reason="handoff", directory=d)
        state, e2 = epoch.claim(holder="b:2", ttl_s=30, directory=d)
        assert e2 == e + 1
        assert state.holder == "b:2"


def test_a_retired_ledger_is_never_adopted() -> None:
    """A dying holder's farewell is a *fresh write*, so mtime alone cannot tell
    it from a real claim.

    2026-11-14: P3 adopted epoch 13 from the previous run's retire(), the real
    claim then moved to 14, and P3 fenced itself within a second -- three boots
    in a row, with a log that looked like an orderly shutdown.
    """
    with tempfile.TemporaryDirectory() as d:
        state, e = epoch.claim(holder="a:1", directory=d)
        assert epoch.adoptable(since_mtime=0.0, directory=d) is not None
        epoch.retire(e, "clean exit", directory=d)

        again = epoch.load(d)
        assert again.retired and again.holder is None
        assert epoch.adoptable(since_mtime=0.0, directory=d) is None, \
            "a retired ledger must never be adopted, however fresh its mtime"
        print(f"       epoch {e} retired -> adoptable() = None")


def test_a_live_claim_is_adopted_and_an_older_one_is_not() -> None:
    with tempfile.TemporaryDirectory() as d:
        epoch.claim(holder="a:1", directory=d)
        got = epoch.adoptable(since_mtime=time.time() - 60.0, directory=d)
        assert got is not None and got.epoch == 1
        assert epoch.adoptable(since_mtime=time.time() + 60.0, directory=d) is None, \
            "a ledger written before this boot belongs to the previous run"
        print("        live claim adopted; a ledger older than the boot is not")


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