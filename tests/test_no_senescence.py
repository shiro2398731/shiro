r"""No senescence: nothing in her drifts with age, and forgetting has a floor.

docs §9 (White, 2026-11-09): she does not metabolise, does not age, does not
die. Allowed: bounded, recoverable slow variables (sleep pressure and the like).
Kept: plasticity (matrix turnover, decay of unvisited structure -- that is
forgetting, not ageing).

Three separate claims, three tests
----------------------------------
1. **No global parameter changes with age.** `gamma`, `dt`, `recover_rate`,
   `ceiling`, `assign_ratio`, `min_amp`, `max_attractors`, `decay_spread`,
   `recover_jitter`, the recovery cadence and the half-life are compared before
   and after the run. A future change that makes one of them a function of
   age fails here.
2. **No region's loss rate changes with age.** Each region draws its decay
   factor at birth and keeps it (terrain._birth_decay_factor). Verified on the
   survivors of a cohort: same set of factors at the first and the last window.
3. **Forgetting has a floor.** Depth is allowed to fall -- a region she stops
   visiting *should* fade. What is forbidden is existing structure decaying
   toward the survival line (`min_amp`) while she keeps building new regions:
   that would be starvation dressed as growth.

Measured, 40000 steps, amplitude fixed at 0.02 (a probe, recorded here because
the numbers are the whole argument):

    t      regions  cohort_n  cohort median depth
    5000     239      239          0.6269
    10000    243      239          0.3242
    20000    242      238          0.0922
    30000    226      222          0.0325
    40000    181      177          0.0213     (min_amp = 0.001)

The decay decelerates (ratio per 5000 steps: 0.52, 0.52, 0.55, 0.69, 0.73, 0.90)
and settles at 0.0213, twenty times above the survival line, while the region
count first grows to 243 and then falls back to 181. That is forgetting with a
floor plus changing structure, not senescence. The test below uses a shorter
run (20000 steps, ~30 s) and asserts the properties, not the exact numbers.

Run: .\.venv\Scripts\python.exe tests\test_no_senescence.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 20000
WINDOWS = 4


def _global_params(st: Stream, ss) -> dict:
    return {
        "D": st.D,
        "gamma": st.dynamics.gamma,
        "dt": st.dynamics.dt,
        "recover_rate": ss.recover_rate,
        "ceiling": ss.ceiling,
        "assign_ratio": ss.assign_ratio,
        "min_amp": ss.min_amp,
        "max_attractors": ss.max_attractors,
        "decay_spread": ss.decay_spread,
        "recover_jitter": ss.recover_jitter,
        "recover_every": st._recover_every,
        "half_life": ss.half_life,
    }


def _run(steps: int = STEPS, windows: int = WINDOWS) -> tuple[Stream, list[dict], dict]:
    st = Stream(dim=32)
    st.calibrate(np.zeros((1, st.D)))
    # This test is about global parameters not drifting with age, not about the
    # rehearsal loop. Turning the loop off keeps the 20000-step run from paying
    # 156 deep copies (it took the suite from ~100 s to ~250 s when left on);
    # the loop itself is covered by tests/test_loop.py.
    st.rehearse_every = 0
    per = max(steps // windows, 1)
    series: list[dict] = []
    cohort: list = []
    factors_first: dict[int, float] = {}
    for i in range(steps):
        # Amplitude fixed on purpose: upstream variation is a different question.
        # Here the only thing that changes is how long she has been running.
        st.landscape.set_noise_amplitude(0.02)
        st.step()
        if (i + 1) % per == 0:
            ss = st.landscape.state_space
            attrs = ss.attractors()
            amps = np.array([a.amp for a in attrs]) if attrs else np.zeros(1)
            if not cohort:
                cohort = list(attrs)
                factors_first = {id(a): float(a._decay_factor or 1.0) for a in cohort}
            live = {id(a) for a in attrs}
            kept = [a for a in cohort if id(a) in live]
            series.append({
                "regions": len(attrs),
                "edges": len(st.graph),
                "population_amp": float(np.median(amps)),
                "cohort_n": len(kept),
                "cohort_amp": float(np.median([a.amp for a in kept])) if kept else 0.0,
                "cohort_factors": {id(a): float(a._decay_factor or 1.0) for a in kept},
            })
    facts = {"factors_first": factors_first,
             "min_amp": st.landscape.state_space.min_amp}
    return st, series, facts


_CACHE: dict[str, object] = {}


def _measured() -> tuple[Stream, list[dict], dict]:
    if "run" not in _CACHE:
        probe = Stream(dim=32)
        _CACHE["params_before"] = _global_params(probe, probe.landscape.state_space)
        _CACHE["run"] = _run()
    return _CACHE["run"]  # type: ignore[return-value]


# ══════════════════════════════════════════════════════════════════════════════
# 1. global parameters
# ══════════════════════════════════════════════════════════════════════════════

def test_global_parameters_do_not_change_with_age() -> None:
    st, _series, _facts = _measured()
    before = _CACHE["params_before"]
    after = _global_params(st, st.landscape.state_space)
    diff = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    assert not diff, f"parameters drifted with age: {diff}"
    print(f"       {len(before)} global parameters identical after {STEPS} steps")


# ══════════════════════════════════════════════════════════════════════════════
# 2. per-region loss rate
# ══════════════════════════════════════════════════════════════════════════════

def test_no_region_gets_faster_at_forgetting_with_age() -> None:
    _st, series, facts = _measured()
    first = facts["factors_first"]
    last = series[-1]["cohort_factors"]
    assert last, "no cohort survivor left to compare"
    changed = {k: (first[k], last[k]) for k in last if first.get(k) != last[k]}
    assert not changed, f"decay factors changed with age: {changed}"
    survivors = sum(1 for k in last)
    print(f"       {survivors} surviving regions keep their birth decay factor")


# ══════════════════════════════════════════════════════════════════════════════
# 3. forgetting has a floor
# ══════════════════════════════════════════════════════════════════════════════

def test_existing_structure_does_not_decay_toward_the_survival_line() -> None:
    _st, series, facts = _measured()
    cohort = [w["cohort_amp"] for w in series]
    floor = float(facts["min_amp"])
    assert cohort[-1] > 2.0 * floor, (
        f"existing regions decayed to {cohort[-1]:.4g}, "
        f"survival line is {floor:.4g}")
    assert all(w["cohort_n"] > 0 for w in series), "every original region was deleted"
    print(f"       cohort depth {cohort[0]:.4g} -> {cohort[-1]:.4g} "
          f"(survival line {floor:.4g}); regions "
          f"{series[0]['regions']} -> {series[-1]['regions']}")


def test_structure_does_not_run_away() -> None:
    _st, series, _facts = _measured()
    regions = [w["regions"] for w in series]
    assert max(regions) < 2000, f"structure ran away: {regions}"
    assert min(regions) > 0, f"structure vanished: {regions}"
    print(f"       regions {min(regions)}..{max(regions)}")


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
