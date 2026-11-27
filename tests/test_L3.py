r"""L3 真测试（2026-11-27）：记忆痕迹不是空表。

原来的 test_L3.py 是 0 字节（src/L3_memory 里也有 6 个 0 字节模块）。
这里钉 TraceGraph 的三件事：边只在「不同区域之间」形成、会衰减、修剪只按存活 uid。

Run: .\.venv\Scripts\python.exe tests\test_L3.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L3_memory.trace import TraceGraph  # noqa: E402


def test_edge_forms_between_different_regions() -> None:
    g = TraceGraph()
    g.touch(1)
    g.touch(2)
    e = g.edge(1, 2)
    assert e is not None and e.w > 0.0, e
    print(f"       edge 1-2 w={e.w:.4f}")


def test_same_region_and_unassigned_do_not_form_edges() -> None:
    g = TraceGraph()
    g.touch(1)
    g.touch(1)          # 同区，不连边
    assert g.edge(1, 1) is None
    g.touch(None)       # 穿过未归属空间，断链
    g.touch(2)
    assert g.edge(1, 2) is None
    print("       same-region / unassigned -> no edge")


def test_weaken_reduces_weight() -> None:
    g = TraceGraph()
    g.touch(1)
    g.touch(2)
    w0 = g.edge(1, 2).w
    g.weaken(rate=0.1)
    w1 = g.edge(1, 2).w if g.edge(1, 2) is not None else 0.0
    assert w1 < w0, (w0, w1)
    print(f"       weaken: w {w0:.4f} -> {w1:.4f}")


def test_prune_only_keeps_alive_uids() -> None:
    g = TraceGraph()
    g.touch(1)
    g.touch(2)
    g.prune({1, 2})
    assert g.edge(1, 2) is not None
    g.prune({1})
    assert g.edge(1, 2) is None
    print("       prune({1,2}) keeps, prune({1}) drops")


def test_roundtrip_keeps_edge() -> None:
    g = TraceGraph()
    g.touch(1)
    g.touch(2)
    g2 = TraceGraph.from_dict(g.to_dict())
    e = g2.edge(1, 2)
    assert e is not None, e
    assert abs(e.w - g.edge(1, 2).w) < 1e-12
    print(f"       roundtrip keeps w={e.w:.4f}")


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
