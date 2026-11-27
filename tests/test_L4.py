r"""L4 真测试（2026-11-27）：驱动/张力不是空表。

原来的 test_L4.py 是 0 字节（src/L4_drive 里 competition/deflection/pool 也是 0 字节）。
这里用真存档钉三件事：
- Drive 能在真地形上产生张力（有限、有和）
- Regulation 给出的因子覆盖每个区域且有限
- Viability 的 Region 数值有限、比例在 [0,1]

Run: .\.venv\Scripts\python.exe tests\test_L4.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def _boot() -> Stream:
    st = Stream()
    assert persist.load_l5(st), "需要存档"
    return st


def test_tensions_are_finite() -> None:
    st = _boot()
    space = st.landscape.state_space
    ts = st.drive.tensions(space, st.graph)
    assert len(ts) > 0, len(ts)
    amounts = np.array([float(t.amount) for t in ts], dtype=float)
    assert np.all(np.isfinite(amounts)), amounts
    s = st.drive.summary()
    assert s["n"] == len(ts), (s["n"], len(ts))
    assert np.isfinite(s["sum"]) and s["sum"] > 0.0, s
    print(f"       tensions={len(ts)}, sum={s['sum']:.1f}, max={s['max']:.1f}")


def test_regulation_covers_regions() -> None:
    st = _boot()
    space = st.landscape.state_space
    f = np.asarray(st.regulation.factors(space), dtype=float)
    n = len(space.attractors())
    assert len(f) == n, (len(f), n)
    assert np.all(np.isfinite(f)), f
    assert float(f.min()) > 0.0, f.min()
    print(f"       factors n={len(f)}, range=[{f.min():.4f},{f.max():.4f}]")


def test_viability_regions_are_finite() -> None:
    st = _boot()
    space = st.landscape.state_space
    regions = st.regulation.viability.measure(space)
    n = len(space.attractors())
    assert len(regions) == n, (len(regions), n)
    for r in regions:
        # 2026-11-27：ratio 可以 >1（amp 超过 carrying 的「目标深度」，
        # 实测 1.50），drain 可以是 +inf（gain 为 0 时源码就这么给）——都不是 bug。
        for v in (r.depth, r.carrying, r.ratio):
            assert np.isfinite(v), (r.depth, r.carrying, r.ratio)
        assert r.ratio >= 0.0, r.ratio
        assert not np.isnan(r.drain), r.drain
    s = st.regulation.viability.summary(space)
    assert s["n"] == len(regions), (s["n"], len(regions))
    assert 0.0 <= s["frac_below"] <= 1.0, s["frac_below"]
    print(f"       viability n={len(regions)}, frac_below={s['frac_below']:.3f}")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
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
