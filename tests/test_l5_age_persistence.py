r"""Age structure and folded mass must survive a restart.

Why this file exists
--------------------
`_born_step` is the denominator of the per-region visit rate, and the visit
rate decides how deep a new region starts (`terrain._birth_depth`). It was not
in the L5 payload. Measured on the real checkpoint before the fix:

    regions 307, total_steps 145504, _born_step distinct values after load: {0}
    _population_visit_rate() = 3.44e-05   (assigned / whole archive life)

i.e. every region came back "as old as the archive", so the measured rate was
two orders of magnitude below the per-region one and new regions were born
below the survival line. The same reasoning applies to `_folded`: the live
table was rebuilt and the summaries were dropped, so folded mass vanished on
every restart.

These tests pin the payload, not the disk: no test here writes to data/state.

Run: .\.venv\Scripts\python.exe tests\test_l5_age_persistence.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import carrier, persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from src.L5_cognition.terrain import Attractor, StateSpace  # noqa: E402


def _aged_stream(n: int = 12, total_steps: int = 40_000) -> Stream:
    """A stream whose regions have distinct ages and distinct visit counts."""
    st = Stream(dim=16)
    ss = st.landscape.state_space
    ss._total_steps = int(total_steps)
    for i in range(n):
        a = Attractor(mu=np.full(ss.D, 0.1 * (i + 1)), amp=0.3)
        a._assigned = int(i + 1)
        a._born_step = int(1000 * i)
        a._decay_factor = 1.0
        ss._attractors.append(a)
    return st


def _born_steps(space: StateSpace) -> list[int]:
    return [int(a._born_step) for a in space.attractors()]


def _folded_mass(space: StateSpace) -> tuple[int, float]:
    return (len(space._folded),
            round(float(sum(f["amp"] for f in space._folded)), 9))


# ══════════════════════════════════════════════════════════════════════════════
# 1. payload round trip (the format her checkpoint and her bundle share)
# ══════════════════════════════════════════════════════════════════════════════

def test_born_step_survives_payload_round_trip() -> None:
    a = _aged_stream()
    before = _born_steps(a.landscape.state_space)
    payload = persist.build_l5_payload(a)

    b = Stream(dim=16)
    persist.apply_l5_payload(b, payload)

    after = _born_steps(b.landscape.state_space)
    assert after == before, f"age lost: before={before} after={after}"
    print(f"       born steps preserved: {after}")


def test_visit_rate_and_birth_depth_are_identical_after_restore() -> None:
    """The two quantities the age structure feeds must not move on a restart."""
    a = _aged_stream()
    ss_a = a.landscape.state_space
    rate_a = ss_a._population_visit_rate()
    depth_a = ss_a._birth_depth(1.0)
    assert rate_a is not None, "need >= 8 regions to measure a population rate"

    b = Stream(dim=16)
    persist.apply_l5_payload(b, persist.build_l5_payload(a))
    ss_b = b.landscape.state_space

    assert ss_b._population_visit_rate() == rate_a, (rate_a, ss_b._population_visit_rate())
    assert ss_b._birth_depth(1.0) == depth_a, (depth_a, ss_b._birth_depth(1.0))
    print(f"       visit_rate={rate_a:.6g}  birth_depth={depth_a:.6g} (unchanged)")


def test_fold_decisions_are_identical_after_restore() -> None:
    a = _aged_stream()
    a.landscape.state_space._total_steps = 40_000
    ids_before = a.landscape.state_space.fold_candidates(min_assigned=2, age_factor=1.0)

    b = Stream(dim=16)
    persist.apply_l5_payload(b, persist.build_l5_payload(a))
    ids_after = b.landscape.state_space.fold_candidates(min_assigned=2, age_factor=1.0)

    assert ids_after == ids_before, f"before={ids_before} after={ids_after}"
    print(f"       fold candidates identical: {len(ids_before)} regions")


def test_folded_summaries_survive_payload_round_trip() -> None:
    a = _aged_stream(n=40)
    ss_a = a.landscape.state_space
    # Stale = rarely assigned and old; the last 5 regions are busy and recent.
    for i, attr in enumerate(ss_a.attractors()):
        attr._assigned = 1 if i < 35 else 40
        attr._born_step = 0 if i < 35 else 39_000
    removed = ss_a.fold(min_assigned=2, age_factor=1.0)
    assert removed == 35, removed
    mass_before = _folded_mass(ss_a)

    b = Stream(dim=16)
    persist.apply_l5_payload(b, persist.build_l5_payload(a))
    ss_b = b.landscape.state_space

    assert len(ss_b.attractors()) == 5, len(ss_b.attractors())
    assert _folded_mass(ss_b) == mass_before, (_folded_mass(ss_b), mass_before)
    print(f"       folded {removed} regions, summaries {mass_before[0]} "
          f"with total amp {mass_before[1]}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. the carrier path (no pickle; what actually moves her to another machine)
# ══════════════════════════════════════════════════════════════════════════════

def test_carrier_keeps_age_and_folded_mass() -> None:
    a = _aged_stream(n=40)
    ss_a = a.landscape.state_space
    for i, attr in enumerate(ss_a.attractors()):
        attr._assigned = 1 if i < 30 else 40
        attr._born_step = 0 if i < 30 else 39_000
    ss_a.fold(min_assigned=2, age_factor=1.0)

    payload = persist.build_l5_payload(a)
    with tempfile.TemporaryDirectory() as d:
        p = carrier.export(payload, pathlib.Path(d) / "her.carrier")
        moved = carrier.load(p, expect_dim=ss_a.D)

    b = Stream(dim=16)
    persist.apply_l5_payload(b, moved)

    assert _born_steps(b.landscape.state_space) == _born_steps(ss_a)
    assert _folded_mass(b.landscape.state_space) == _folded_mass(ss_a)
    print(f"       carrier: born steps + {_folded_mass(b.landscape.state_space)[0]} "
          f"summaries intact")


# ══════════════════════════════════════════════════════════════════════════════
# 3. backward compatibility: an older checkpoint has no `_born_step` key
# ══════════════════════════════════════════════════════════════════════════════

def test_legacy_payload_without_born_step_still_loads() -> None:
    a = _aged_stream()
    payload = persist.build_l5_payload(a)
    for d in payload["attractors"]:
        d.pop("_born_step")
    # 2026-11-17：摘要不再进存档（build_l5_payload 已经没有这个键）。
    # 这个测试本来是想验证「旧存档没有 folded 也能载入」，所以用 pop 的默认值。
    payload.pop("folded", None)

    b = Stream(dim=16)
    persist.apply_l5_payload(b, payload)

    assert _born_steps(b.landscape.state_space) == [0] * 12, "legacy default must be 0"
    assert b.landscape.state_space._folded == []
    print("       legacy payload: 12 regions, born_step defaulted to 0, no summaries")


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
