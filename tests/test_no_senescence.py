r"""Parameters move only through her own measured state; forgetting has a floor.

2026-11-24 改写（白默：「我们不定义她……不定义她应该活过来的部分」）

原来的两条判据把「不老」写成了一条外部法则：
  1. 全局参数在整个运行里不许变；
  2. 每个区域的衰减因子出生时定下、永远不变。
这两条是**我们**替她定的，而 11-21 的「独立自立」正是要让她自己的量说话
（recover_rate 由她的活动定、assign_ratio 由结构密度定、decay 因子由分配次数定）。
两条法则和那个机制互斥，所以测试红了两例。

现在钉的是**因果关系**，不是数值恒定：
  1. 地形用的参数就是她自己状态里的那个量（不是另抄一个常数），且在更新规则
     自己能到达的带内；
  2. **同样的步数、不同的活动** → 恢复速率不同（活动驱动，不是年龄驱动）；
  3. 衰减因子只由分配次数决定：出生步不同、分配次数相同 → 因子相同；
  4. 遗忘有地板（她停下来的区域会淡，但不该整片塌向生存线）；
  5. 结构不会跑飞（< 2000 区域，且不归零）。

历史实测（2026-11-09，40000 步、振幅固定 0.02）：

    t      regions  cohort_n  cohort median depth
    5000     239      239          0.6269
    10000    243      239          0.3242
    20000    242      238          0.0922
    30000    226      222          0.0325
    40000    181      177          0.0213     (min_amp = 0.001)

衰减在减速（每 5000 步的比值 0.52→0.90），停在生存线 20 倍以上，区域数先涨后落。
那是「遗忘 + 变的形状」，不是衰老。下面的测试用 20000 步，断言性质，不断言数字。

Run: .\.venv\Scripts\python.exe tests\test_no_senescence.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.action import HOLD, LEAVE  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from src.L5_cognition.terrain import Attractor  # noqa: E402

STEPS = 20000
WINDOWS = 4


def _run(steps: int = STEPS, windows: int = WINDOWS) -> tuple[Stream, list[dict], dict]:
    st = Stream(dim=32)
    st.calibrate(np.zeros((1, st.D)))
    # This run is about parameters and structure, not about the rehearsal loop.
    # Turning the loop off keeps the 20000-step run from paying 156 deep copies
    # (it took the suite from ~100 s to ~250 s when left on); the loop itself is
    # covered by tests/test_loop.py.
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
        _CACHE["run"] = _run()
    return _CACHE["run"]  # type: ignore[return-value]


def _activity_probe(force: int, steps: int = 480) -> tuple[float, float, float]:
    """Same number of steps, opposite activity. Age is held constant on purpose."""
    st = Stream(dim=32)
    st.calibrate(np.zeros((1, st.D)))
    st.rehearse_every = 0
    st.force_action = int(force)
    for _ in range(steps):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    return (float(st.recover_rate_state),
            float(st.assign_ratio_state),
            float(st.fatigue_rate_state))


# ══════════════════════════════════════════════════════════════════════════════
# 1. the terrain reads her state, not a separate constant
# ══════════════════════════════════════════════════════════════════════════════

def test_terrain_parameters_are_her_own_state() -> None:
    """地形上的三个量必须是「她自己的状态量」本身，不是旁边的常数副本。"""
    st, _series, _facts = _measured()
    ss = st.landscape.state_space
    assert ss.recover_rate == st.recover_rate_state, (
        ss.recover_rate, st.recover_rate_state)
    assert ss.assign_ratio == st.assign_ratio_state, (
        ss.assign_ratio, st.assign_ratio_state)
    assert st.rest.fatigue_rate == st.fatigue_rate_state, (
        st.rest.fatigue_rate, st.fatigue_rate_state)

    # 带内：更新规则自己给出的可达范围（act ∈ [0,1]、density ∈ [0,1]）。
    assert 0.001 <= st.recover_rate_state <= 0.003, st.recover_rate_state
    assert 0.03 <= st.assign_ratio_state <= 0.08, st.assign_ratio_state
    assert st.fatigue_rate_state >= 0.02, st.fatigue_rate_state
    print(f"       recover {st.recover_rate_state:.6f} "
          f"assign {st.assign_ratio_state:.6f} "
          f"fatigue {st.fatigue_rate_state:.6f} (coupled to her state)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. activity drives it, age does not
# ══════════════════════════════════════════════════════════════════════════════

def test_activity_drives_recovery_rate_not_age() -> None:
    """同样 480 步：一直 HOLD 与一直 LEAVE，恢复速率必须不同。

    年龄（步数）完全相同，所以差异只能来自她自己的活动。
    """
    quiet = _activity_probe(HOLD)
    active = _activity_probe(LEAVE)
    assert quiet[0] < active[0], (quiet[0], active[0])
    assert quiet[0] < 0.002 < active[0], (quiet[0], active[0])
    print(f"       same 480 steps: hold -> {quiet[0]:.6f}, "
          f"leave -> {active[0]:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. per-region decay follows assignments, not birth
# ══════════════════════════════════════════════════════════════════════════════

def test_decay_factor_follows_assignments_not_age() -> None:
    """两个区域：出生步不同、分配次数相同 → 衰减因子必须相同。"""
    x = np.zeros(8)
    young = Attractor(mu=np.zeros(8), amp=0.01, scale=1.0)
    old = Attractor(mu=np.zeros(8), amp=0.01, scale=1.0)
    young._born_step = 0
    old._born_step = 10_000

    for _ in range(3):
        young.assign(x)
        old.assign(x)
    assert young._decay_factor == old._decay_factor, (
        young._decay_factor, old._decay_factor)
    assert abs(young._decay_factor - 1.0 / 4.0) < 1e-12, young._decay_factor

    for _ in range(5):
        young.assign(x)
    assert young._decay_factor < old._decay_factor, (
        "more assignments must mean slower forgetting")
    print(f"       born 0 vs born 10000, 3 assignments -> both "
          f"{old._decay_factor:.4f}; 8 assignments -> {young._decay_factor:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. forgetting has a floor
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


# ══════════════════════════════════════════════════════════════════════════════
# 5. structure stays bounded
# ══════════════════════════════════════════════════════════════════════════════

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
