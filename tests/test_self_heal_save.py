r"""save_with_self_heal 的测试：撞上限时真的会自己归一、重试、存下来。

2026-11-19 上午：这段逻辑原来内联在 P2 主循环里，等于**永远不会被单独验证**。
抽成函数后才有这个测试。

**安全**：测试把 `persist.STATE_DIR` 指到临时目录，绝不碰运行中的 `data/state/l5.pkl`。

Run: .\.venv\Scripts\python.exe tests\test_self_heal_save.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.reorigin import max_component  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from src.runtime.process_2_mind import save_with_self_heal  # noqa: E402


class FakeObserver:
    def __init__(self) -> None:
        self.events = []

    def event(self, name, payload=None, **kw):  # noqa: ANNOT001
        self.events.append((name, payload or kw))


def _load_stream():
    st = Stream()
    if not persist.load_l5(st):
        return None
    return st


def test_normal_save_does_not_shift() -> None:
    st = _load_stream()
    if st is None:
        print("       skipped: no archive")
        return
    with tempfile.TemporaryDirectory() as td:
        old = persist.STATE_DIR
        persist.STATE_DIR = pathlib.Path(td)
        try:
            ob = FakeObserver()
            shifted, info = save_with_self_heal(st, ob)
            assert shifted is False, (shifted, info)
            assert info is None
            assert (pathlib.Path(td) / "l5.pkl").exists()
            assert ob.events == []
        finally:
            persist.STATE_DIR = old
    print("       normal path: saves once, no shift, no event")


def test_overflow_shifts_and_saves() -> None:
    st = _load_stream()
    if st is None:
        print("       skipped: no archive")
        return
    # 把整片地形和她一起推过 1e6（真实场景：她走远了，区域跟着走远）
    for a in st.landscape.state_space.attractors():
        a.mu = np.asarray(a.mu, dtype=float) + 1.5e6
    st._st.x = np.asarray(st._st.x, dtype=float) + 1.5e6
    before = max_component(st)
    assert before > 1e6, before

    with tempfile.TemporaryDirectory() as td:
        old = persist.STATE_DIR
        persist.STATE_DIR = pathlib.Path(td)
        try:
            ob = FakeObserver()
            shifted, info = save_with_self_heal(st, ob)
            assert shifted is True, (shifted, info)
            assert info is not None
            assert info["after"] < info["before"], info
            assert (pathlib.Path(td) / "l5.pkl").exists()
            names = [n for n, _ in ob.events]
            assert "reorigin_persist" in names, names
            # 搬完必须能存下来（也就是降到了上限以下）
            assert max_component(st) < 1e6
        finally:
            persist.STATE_DIR = old
    print(f"       overflow: {before:,.0f} -> {info['after']:,.0f}, "
          f"moved={info['moved']}, event recorded")


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
