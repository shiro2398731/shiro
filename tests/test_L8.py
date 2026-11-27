r"""L8 真测试（2026-11-27）：身份层不是空壳。

原来的 test_L8.py 是 0 字节（src/L8_identity 里 causal_chain/compression/reference/
self_model 也是 0 字节）。这里钉：Identity 能从不大的痕迹图里提核心、能修剪、能出指纹。

Run: .\.venv\Scripts\python.exe tests\test_L8.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L3_memory.trace import TraceGraph  # noqa: E402
from src.L8_identity.identity import Identity  # noqa: E402


def _graph() -> TraceGraph:
    g = TraceGraph()
    for node in (1, 2, 2, 3, 1):
        g.touch(node)
    return g


def test_observe_builds_core() -> None:
    g = _graph()
    ident = Identity()
    core = ident.observe(g)
    assert not core.is_empty(), core.describe()
    assert core.nodes(), core.nodes()
    assert ident.rounds == 1, ident.rounds
    assert "edges" in core.describe(), core.describe()
    structural = ident.structural(1)
    assert not structural.is_empty(), structural.describe()
    print(f"       core={core.describe()}, rounds={ident.rounds}")


def test_prune_returns_count() -> None:
    g = _graph()
    ident = Identity()
    ident.observe(g)
    n = ident.prune({1})
    assert isinstance(n, int) and n >= 0, n
    print(f"       prune({1}) -> {n}")


def test_fingerprint_is_dict() -> None:
    from src.common import persist
    from src.L5_cognition.stream import Stream

    st = Stream()
    assert persist.load_l5(st), "需要存档"
    fp = Identity().fingerprint(st.landscape.state_space, st.graph)
    assert isinstance(fp, dict) and fp, type(fp)
    print(f"       fingerprint keys={sorted(fp)[:5]}")


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
