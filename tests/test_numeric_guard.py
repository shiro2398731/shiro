r"""A blown-up shape must never become a checkpoint.

2026-11-08: a runaway during testing wrote 617 regions with centroids at 1e306,
then nan. Every later load inherited it and the archive had to be restored from
a backup. These guards are the reason that cannot happen silently again.

Run: .\.venv\Scripts\python.exe tests\test_numeric_guard.py
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

from src.common import persist  # noqa: E402
from src.L5_cognition.terrain import StateSpace  # noqa: E402


def _payload_with(mu_value: float, amp: float = 0.5) -> dict:
    return {
        "attractors": [{"mu": np.full(StateSpace().D, mu_value, dtype=float), "amp": amp,
                        "members": [], "scale": 1.0, "born_t": None,
                        "_decay_factor": 1.0, "_assigned": 1}],
        "graph": {}, "identity": {"rounds": 0, "appear": {}},
        "zero": np.zeros(StateSpace().D), "x": None, "v": None,
        "step_count": 0, "total_steps": 0, "global_scale": 1.0, "recent": [],
    }


def test_finite_payload_passes() -> None:
    persist.assert_l5_finite(_payload_with(0.5))


def test_nan_centroid_is_refused() -> None:
    try:
        persist.assert_l5_finite(_payload_with(float("nan")))
        raise AssertionError("nan centroid accepted")
    except ValueError as e:
        print(f"       refused: {e}")


def test_infinite_centroid_is_refused() -> None:
    try:
        persist.assert_l5_finite(_payload_with(1e306))  # finite but absurd
        raise AssertionError("1e306 centroid accepted")
    except ValueError as e:
        print(f"       refused: {e}")


def test_non_finite_observation_is_refused() -> None:
    space = StateSpace()
    bad = np.zeros(space.D)
    bad[0] = np.inf
    try:
        space.record(bad)
        raise AssertionError("non-finite observation recorded")
    except ValueError as e:
        print(f"       refused: {e}")


def test_save_refuses_and_leaves_previous_file_intact() -> None:
    """The whole point: a run that goes bad must not damage the good checkpoint.

    Uses a real Stream rather than a stub. A stub had to be extended every time
    the payload grew a field (self_model, then grounding), and each extension
    was invisible until the test failed: the payload contract is exactly what
    this test should be exercising, not bypassing.
    """
    from src.L5_cognition.stream import Stream

    with tempfile.TemporaryDirectory() as d:
        old = persist.STATE_DIR
        persist.STATE_DIR = pathlib.Path(d)
        try:
            stream = Stream(dim=32)
            space = stream.landscape.state_space
            from src.L5_cognition.terrain import Attractor

            space._attractors.append(Attractor(mu=np.zeros(space.D), amp=0.5))
            space._attractors.append(Attractor(mu=np.ones(space.D) * 0.1, amp=0.3))

            persist.save_l5(stream, "l5.pkl")
            before = (pathlib.Path(d) / "l5.pkl").read_bytes()

            # now poison the space
            space._attractors[0].mu = np.full(space.D, np.nan)
            try:
                persist.save_l5(stream, "l5.pkl")
                raise AssertionError("poisoned shape was checkpointed")
            except ValueError:
                pass

            assert (pathlib.Path(d) / "l5.pkl").read_bytes() == before
            print("       poisoned shape blocked, previous checkpoint intact")
        finally:
            persist.STATE_DIR = old


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