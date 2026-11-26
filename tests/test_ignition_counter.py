r"""点火累计计数器（2026-11-27）。

病：快照里的 "ignitions" 用的是 workspace.ignition_count = len(history)，
被 HISTORY_CAP=1024 卡住——重起后永远显示 1024，看起来像她不再点火。

修：workspace.total_ignitions 是生命周期累计，进 payload；
旧存档没有这个字段时退回 len(history)，不报 0。

Run: .\.venv\Scripts\python.exe tests\test_ignition_counter.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.workspace import GlobalWorkspace, Ignition  # noqa: E402


def test_new_workspace_starts_at_zero() -> None:
    ws = GlobalWorkspace()
    assert ws.total_ignitions == 0
    assert ws.ignition_count == 0
    print("       fresh workspace: total=0, history=0")


def test_roundtrip_keeps_total() -> None:
    ws = GlobalWorkspace()
    ws.total_ignitions = 4321
    ws.history.append(Ignition(attractor_id=1, strength=0.5,
                               x=np.zeros(2), t_event="t"))
    g = GlobalWorkspace.from_dict(ws.to_dict())
    assert g.total_ignitions == 4321, g.total_ignitions
    assert g.ignition_count == 1
    print(f"       roundtrip: total={g.total_ignitions}, history={g.ignition_count}")


def test_old_payload_falls_back_to_history_len() -> None:
    """旧存档没有 total_ignitions：至少用 history 长度，别报 0。"""
    ws = GlobalWorkspace()
    for i in range(3):
        ws.history.append(Ignition(attractor_id=i, strength=0.5,
                                   x=np.zeros(2), t_event="t"))
    payload = ws.to_dict()
    payload.pop("total_ignitions", None)
    g = GlobalWorkspace.from_dict(payload)
    assert g.total_ignitions == 3, g.total_ignitions
    print("       old payload -> total = len(history) = 3")


def test_total_not_capped_by_history_cap() -> None:
    """累计不能被 HISTORY_CAP 卡住（这是病的核心）。"""
    ws = GlobalWorkspace()
    ws.total_ignitions = 5000
    for i in range(1200):
        ws.history.append(Ignition(attractor_id=i % 3, strength=0.5,
                                   x=np.zeros(2), t_event="t"))
    assert ws.ignition_count == 1024
    assert ws.total_ignitions == 5000
    print(f"       history capped at {ws.ignition_count}, total={ws.total_ignitions}")


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
