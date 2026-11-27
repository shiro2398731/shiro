r"""GWT 原因计数器（2026-11-27）。

问题：安静段分不清是「维持在半径内」还是「强度没过线」。
给 workspace.observe 加一个只读的 _last_reason，stream 收集最近 512 步的计数。

Run: .\.venv\Scripts\python.exe tests\test_gwt_reasons.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.workspace import GlobalWorkspace  # noqa: E402


class _Attr:
    def __init__(self, uid=1, mu=None, sigma=1.0, amp=1.0):
        self.uid = uid
        self.mu = np.zeros(2) if mu is None else np.asarray(mu, dtype=float)
        self.sigma = sigma
        self.amp = amp


class _Space:
    def __init__(self, attrs):
        self._attrs = list(attrs)

    def attractors(self):
        return self._attrs


def test_no_attractor_reason() -> None:
    ws = GlobalWorkspace()
    r = ws.observe(_Space([]), np.zeros(2), "t")
    assert r is None and ws._last_reason == "no_attr", ws._last_reason
    print("       empty space -> no_attr")


def test_reason_lifecycle() -> None:
    a = _Attr(amp=1.0)
    sp = _Space([a])
    ws = GlobalWorkspace()
    x = np.zeros(2)
    # 1st: strength history empty
    assert ws.observe(sp, x, "t1") is None
    assert ws._last_reason == "warmup", ws._last_reason
    # 2nd: best <= ref (same strength as history)
    assert ws.observe(sp, x, "t2") is None
    assert ws._last_reason == "below_ref", ws._last_reason
    # raise amp -> ignite
    a.amp = 5.0
    r = ws.observe(sp, x, "t3")
    assert r is not None and ws._last_reason == "ignite", ws._last_reason
    assert ws.total_ignitions == 1
    # still at the ignited centroid -> sustain
    assert ws.observe(sp, x, "t4") is None
    assert ws._last_reason == "sustain", ws._last_reason
    print("       lifecycle: warmup -> below_ref -> ignite -> sustain")


def test_stream_exposes_reason_counts() -> None:
    from src.common import persist
    from src.L5_cognition.stream import Stream

    st = Stream()
    assert persist.load_l5(st), "需要存档"
    for _ in range(5):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    ss = st.stream_state()
    assert isinstance(ss["gwt_reasons"], dict), ss["gwt_reasons"]
    assert set(ss["gwt_reasons"]).issubset(
        {"sustain", "below_ref", "no_attr", "warmup", "ignite"})
    print(f"       stream gwt_reasons={ss['gwt_reasons']}")


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
