r"""全局工作空间 / 随机数的存档往返。

2026-11-24 夜：复核发现两个缺口，这里把它们钉住。

1. `workspace` 不进存档 → 重启后 `_strength_hist` 空、`reference_level()` 为
   None、256 步内不可能点火；点火记录也全丢，「她注意什么」在重启前后无法对照。
2. `born_seed` 存了却没人用它播种 → 同一份存档读两次，rng 不同，轨迹不同
   （实测第一次正态抽样 -1.570 vs 1.684）。`test_uid` 的重命名测试因此红。
   现在存的是 `bit_generator.state`：重启是**接着**她的骰子，不是从头再掷。

Run: .\.venv\Scripts\python.exe tests\test_workspace_persist.py
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

_CACHE: dict = {}


def _payload() -> dict:
    if "p" not in _CACHE:
        st = Stream()
        if not persist.load_l5(st):
            raise SystemExit("no L5 archive for this test")
        for _ in range(6):
            st.landscape.set_noise_amplitude(0.02)
            st.step()
        _CACHE["p"] = persist.build_l5_payload(st)
        _CACHE["ws"] = st.workspace.to_dict()
        _CACHE["rng_state"] = copy.deepcopy(st.rng.bit_generator.state)
    return _CACHE["p"]


def _apply(payload: dict) -> Stream:
    st = Stream()
    persist.apply_l5_payload(st, payload)
    return st


def test_workspace_survives_round_trip() -> None:
    before = _CACHE["ws"]
    st = _apply(_payload())
    after = st.workspace.to_dict()
    assert after["strength_hist"] == before["strength_hist"], (
        len(after["strength_hist"]), len(before["strength_hist"]))
    assert len(after["history"]) == len(before["history"])
    assert after["current"] == before["current"]
    assert after["active_sigma"] == before["active_sigma"]
    print(f"       workspace: {len(after['strength_hist'])} peaks, "
          f"{len(after['history'])} ignitions restored")


def test_reference_level_is_defined_after_load() -> None:
    st = _apply(_payload())
    if len(_CACHE["ws"]["strength_hist"]) < 1:
        raise SystemExit("archive has no peak history to restore")
    assert st.workspace.reference_level() is not None, (
        "no reference level after load -> 256 steps of guaranteed no ignition")
    print(f"       reference level after load = "
          f"{st.workspace.reference_level():.6f}")


def test_two_loads_draw_the_same_numbers() -> None:
    payload = _payload()
    a = _apply(copy.deepcopy(payload))
    b = _apply(copy.deepcopy(payload))
    da, db = float(a.rng.normal()), float(b.rng.normal())
    assert da == db, (da, db)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    print(f"       two loads, same first draw = {da:.6f}")


def test_two_loads_walk_the_same_trajectory() -> None:
    payload = _payload()
    a = _apply(copy.deepcopy(payload))
    b = _apply(copy.deepcopy(payload))
    for _ in range(20):
        for st in (a, b):
            st.landscape.set_noise_amplitude(0.02)
            st.step()
    assert a._st.x.tobytes() == b._st.x.tobytes(), (
        "same checkpoint, different trajectory: rng state not restored")
    print("       20 steps from the same checkpoint -> identical position")


def test_rng_state_survives_json() -> None:
    payload = _payload()
    raw = json.dumps(payload["rng_state"])
    assert json.loads(raw) == payload["rng_state"], "rng state not JSON safe"
    print("       rng_state is JSON safe (carrier path)")


def test_old_payload_without_workspace_still_loads() -> None:
    payload = copy.deepcopy(_payload())
    payload.pop("workspace", None)
    payload.pop("rng_state", None)
    st = _apply(payload)
    assert st.workspace.ignition_count == 0
    assert st.workspace.reference_level() is None
    assert np.isfinite(st.rng.normal())
    print("       old checkpoint: empty workspace, fresh rng, no crash")


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
