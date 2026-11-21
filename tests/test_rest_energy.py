"""生命层两个机制的测试：rest（趋性/疲劳）+ energy（有限性/休眠）。

钉住的是「机制」不是「数值」：
    rest  = 占据时 u 上升、不占据时 u 下降、有效深度变浅
    energy= 活动才耗、耗尽变 asleep、睡着不动、存档往返

Run: .\.venv\Scripts\python.exe tests\test_rest_energy.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.rest import RestField  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


# ---------- rest（疲劳场） ----------

def test_rest_touch_tracks() -> None:
    f = RestField()
    f.touch(0)
    f.touch(1)
    print("      touch 后注册了 2 个区域")


def test_rest_u_rises_when_occupied() -> None:
    """占据时 u 上升（待久了变浅）。"""
    f = RestField()
    for i in range(3):
        f.touch(i)
    u0 = f.value(0)
    for _ in range(20):
        f.step(occupied=0)
    u1 = f.value(0)
    assert u1 > u0, (u0, u1)
    print("      占据 20 步: u %.3f -> %.3f（上升）" % (u0, u1))


def test_rest_u_falls_when_unoccupied() -> None:
    """不占据时 u 下降（恢复）。"""
    f = RestField()
    for i in range(3):
        f.touch(i)
    for _ in range(20):
        f.step(occupied=0)      # 先把它占据到较高
    u_high = f.value(0)
    for _ in range(200):
        f.step(occupied=1)      # 不再占据它
    u_low = f.value(0)
    assert u_low < u_high, (u_high, u_low)
    print("      不占据 200 步: u %.3f -> %.3f（下降）" % (u_high, u_low))


def test_rest_u_bounded() -> None:
    """u 有界 [0,1]。"""
    f = RestField()
    f.touch(0)
    for _ in range(5000):
        f.step(occupied=0)
    assert 0.0 <= f.value(0) <= 1.0 + 1e-9, f.value(0)
    print("      u 有界：%.3f" % f.value(0))


def test_rest_to_dict_roundtrip() -> None:
    """to_dict/from_dict 往返。"""
    f = RestField()
    for i in range(5):
        f.touch(i)
    for _ in range(30):
        f.step(occupied=2)
    d = f.to_dict()
    g = RestField.from_dict(d)
    assert abs(f.value(2) - g.value(2)) < 1e-12, (f.value(2), g.value(2))
    print("      存档往返: u=%.4f" % g.value(2))


# ---------- energy（有限性） ----------

def test_energy_activity_based() -> None:
    """活动（leave）时耗得多，休息（hold）时耗得少。"""
    st = Stream()
    assert st.energy == 1.0, st.energy
    assert st.energy_burn > 0, st.energy_burn
    assert not st.asleep, st.asleep
    print("      energy=%.3f burn=%.1e asleep=%s" % (st.energy, st.energy_burn, st.asleep))


def test_energy_exhaustion_sleeps() -> None:
    """energy 见底 -> asleep=True；睡着时 step 不动。

    2026-11-21：加了「补充」之后，见底要同时「没有电」才睡得着——
    所以这里把 recharge 设 0（模拟断电），否则补充会把 energy 拉回来。
    """
    from src.common import persist
    st = Stream()
    assert persist.load_l5(st), "需要存档（加载失败）"
    st.energy = 1e-9
    st.energy_recharge = 0.0
    st.landscape.set_noise_amplitude(0.02)
    st.step()  # 见底
    assert st.asleep, st.asleep
    e = st.energy
    r = st.step()  # 睡着
    assert r is None, r
    assert st.energy == e, (st.energy, e)
    print("      见底: asleep=%s，再 step 返回 None 且 energy 不变" % st.asleep)


def test_energy_does_not_kill() -> None:
    """耗尽只休眠，不改变区域数（不是死）。

    2026-11-21：recover_rate 现在是状态变量（每 16 步随她的活动调），
    所以步进后区域数可能动几格——这里放容差 ±10，钉的是「没有整个消失」。
    """
    from src.common import persist
    st = Stream()
    assert persist.load_l5(st), "需要存档（加载失败）"
    assert st.landscape.state_space.attractors(), "需要存档"
    n0 = len(st.landscape.state_space.attractors())
    st.energy = 1e-9
    st.energy_recharge = 0.0
    st.landscape.set_noise_amplitude(0.02)
    st.step()
    assert st.asleep
    n1 = len(st.landscape.state_space.attractors())
    assert abs(n1 - n0) <= 10, (n0, n1)
    print("      耗尽不影响区域数（%d -> %d）：不是死，是休眠" % (n0, n1))


def test_energy_save_roundtrip() -> None:
    """energy / asleep 进存档。"""
    from src.common import persist
    st = Stream()
    if not persist.load_l5(st):
        print("      跳过（没有存档）")
        return
    st.energy = 0.4321
    st.asleep = False
    payload = persist.build_l5_payload(st)
    assert "energy" in payload, "payload 缺 energy"
    assert "asleep" in payload, "payload 缺 asleep"
    st2 = Stream()
    persist.apply_l5_payload(st2, payload)
    assert abs(st2.energy - 0.4321) < 1e-9, st2.energy
    print("      存档往返: energy=%.4f" % st2.energy)


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
