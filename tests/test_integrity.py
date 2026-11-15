r"""一致性检查本身要被检查。

今晚同一族错误出现了四次：某个地方存着区域编号，而表的形状变了。症状是"数字看起来正常、
行为悄悄错"。`tools/integrity.py` 就是为这一类准备的，所以它必须**真的会叫**。

这里做五件事：干净的 Stream 不报错，以及四类注入的损坏各自被指名：
越界编号、中毒质心、计数对不上、图边指向表外。

Run: .\.venv\Scripts\python.exe tests\test_integrity.py
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.L5_cognition.stream import Stream  # noqa: E402
from tools.integrity import check  # noqa: E402


def _ran(steps: int = 30) -> Stream:
    st = Stream(dim=16)
    st.calibrate(np.zeros((1, 16)))
    for _ in range(steps):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    return st


def test_a_healthy_stream_reports_nothing() -> None:
    st = _ran()
    problems = check(st, journal_path=None)
    assert problems == [], problems
    print(f"       {len(st.landscape.state_space.attractors())} regions, no problems")


def test_an_out_of_range_symbol_is_reported() -> None:
    st = _ran()
    n = len(st.landscape.state_space.attractors())
    st.grounding.observe(n + 7, (0, 0))
    problems = check(st, journal_path=None)
    assert any("grounding.symbol" in p for p in problems), problems
    print(f"       injected symbol {n + 7} -> " + [p for p in problems if "grounding" in p][0])


def test_a_poisoned_centroid_is_reported() -> None:
    st = _ran()
    st.landscape.state_space.attractors()[0].mu = np.full(16, 1e300)
    problems = check(st, journal_path=None)
    assert any("centroid" in p for p in problems), problems
    print("       injected |mu|=1e300 -> reported")


def test_counts_that_do_not_add_up_are_reported() -> None:
    st = _ran()
    st.grounding.observe(0, (1, 1))
    st.grounding.total += 99
    problems = check(st, journal_path=None)
    assert any("grounding.total" in p for p in problems), problems
    print("       injected total mismatch -> reported")


def test_a_graph_edge_pointing_at_a_dead_region_is_reported() -> None:
    st = _ran()
    space = st.landscape.state_space
    graph = st.graph
    # a uid that no region has: the shape a stale node takes after a fold
    dead = max(space.alive_uids()) + 7
    key = (1, dead)
    try:
        from src.L3_memory.trace import Edge
        graph._edges[key] = Edge(a=key[0], b=key[1])
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"could not inject the edge: {e}")
    problems = check(st, journal_path=None)
    assert any("graph.edge.b" in p for p in problems), problems
    print(f"       injected edge (1, {dead}) -> reported")


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
