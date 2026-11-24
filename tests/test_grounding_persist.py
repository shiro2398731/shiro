r"""指代分桶历史的存档往返（C1：经验连续性）。

2026-11-24：`GroundingTable.to_dict` 只存 counts，不存 `_hist` ——「她自己历史上
的速度/张力分布」。重启后 `bucket()` 在 64 个样本之前一律 None，于是指代标签
是 "none"。这不是「还没开始」，是「她又忘了自己平常长什么样」。

Run: .\.venv\Scripts\python.exe tests\test_grounding_persist.py
"""

from __future__ import annotations

import copy
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 90  # > MIN_HIST (64)，让分桶历史真的有用


def _filled() -> Stream:
    st = Stream(dim=32)
    st.calibrate(np.zeros((1, st.D)))
    st.rehearse_every = 0
    for _ in range(STEPS):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    return st


_CACHE: dict = {}


def _payload() -> dict:
    if "p" not in _CACHE:
        st = _filled()
        _CACHE["st"] = st
        _CACHE["p"] = persist.build_l5_payload(st)
    return _CACHE["p"]


def test_histories_are_in_the_payload() -> None:
    p = _payload()
    qh = p["grounding"].get("quantile_hist") or {}
    assert qh, "payload has no quantile_hist"
    for name, vals in qh.items():
        assert len(vals) > 0, name
    print(f"       payload carries histories for {sorted(qh)} "
          f"(lens {[len(v) for v in qh.values()]})")


def test_round_trip_keeps_them() -> None:
    st = _CACHE.get("st") or _filled()
    before = {k: [float(v) for v in dq] for k, dq in st.grounding._hist.items()}
    st2 = Stream(dim=32)
    persist.apply_l5_payload(st2, copy.deepcopy(_payload()))
    after = {k: [float(v) for v in dq] for k, dq in st2.grounding._hist.items()}
    assert set(after) == set(before), (sorted(before), sorted(after))
    for k in before:
        assert after[k] == before[k], k
    print(f"       {len(before)} histories identical after round trip")


def test_buckets_still_work_after_reload() -> None:
    st = _CACHE.get("st") or _filled()
    st2 = Stream(dim=32)
    persist.apply_l5_payload(st2, copy.deepcopy(_payload()))
    v = 0.37
    a = st.grounding.bucket("speed", v)
    b = st2.grounding.bucket("speed", v)
    assert a is not None, "history too short in the source stream"
    assert b == a, (a, b)
    print(f"       bucket(speed, {v}) = {b} on both sides")


def test_old_payload_without_histories_is_honest() -> None:
    p = copy.deepcopy(_payload())
    p["grounding"].pop("quantile_hist", None)
    st = Stream(dim=32)
    persist.apply_l5_payload(st, p)
    assert not st.grounding._hist, "a legacy table must not invent histories"
    assert st.grounding.bucket("speed", 0.37) is None
    print("       legacy checkpoint: no history, bucket = None (not fabricated)")


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
