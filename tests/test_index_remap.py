r"""区域编号在结构变化后必须跟着走。

区域编号是**位置性**的：删掉中间一个，后面每个都往前挪一位。而上下游都存着编号
（图、指代表、自模型、广播给 L6 的 id、预演的预测序列）。不跟着改，它们就会静默指到
邻居身上——这是"看起来有数据"的第三类来源。

`StateSpace` 现在在每次结构变化（recover 删除 / 合并 / 折叠）后发布一张 old->new 表，
`Stream` 取走它，把身上所有编号改一遍，并把同一张表交给 `Consolidator.sync`。

Run: .\.venv\Scripts\python.exe tests\test_index_remap.py
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

from src.L3_memory.consolidate import Consolidator  # noqa: E402
from src.L3_memory.trace import TraceGraph  # noqa: E402
from src.L5_cognition.grounding import GroundingTable  # noqa: E402
from src.L5_cognition.self_model import OnlineSelfModel  # noqa: E402
from src.L5_cognition.terrain import Attractor, StateSpace  # noqa: E402


def _space(n: int = 4, dim: int = 8) -> StateSpace:
    ss = StateSpace(dim=dim)
    for i in range(n):
        ss._attractors.append(Attractor(mu=np.full(dim, 0.3 * i), amp=0.5))
    ss._total_steps = 100
    return ss


# ══════════════════════════════════════════════════════════════════════════════
# 1. 状态空间发布正确的一张表
# ══════════════════════════════════════════════════════════════════════════════

def test_recover_publishes_old_to_new_and_clears_it() -> None:
    ss = _space(4)
    ss._attractors[1].amp = 1e-9          # will fall below min_amp
    ss.recover()
    assert len(ss.attractors()) == 3
    assert ss.take_index_remap() == {0: 0, 2: 1, 3: 2}
    assert ss.take_index_remap() is None, "the map must be handed out once"
    print("       removing index 1 -> {0:0, 2:1, 3:2}")


def test_no_remap_when_nothing_was_removed() -> None:
    ss = _space(4)
    ss.recover()
    assert len(ss.attractors()) == 4
    assert ss.take_index_remap() is None
    print("       a clean recover publishes nothing")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 图跟着走（旧行为会留下指错的边）
# ══════════════════════════════════════════════════════════════════════════════

def test_graph_edges_follow_a_middle_removal() -> None:
    ss = _space(4)
    g = TraceGraph()
    for node in (0, 1, 2, 3):
        g.touch(node)
    edges_before = {(e.a, e.b) for e in g.edges()}
    assert edges_before == {(0, 1), (1, 2), (2, 3)}, edges_before

    ss._attractors[1].amp = 1e-9
    ss.recover()
    remap = ss.take_index_remap()
    Consolidator().sync(g, ss, remap=remap)

    edges_after = {(e.a, e.b) for e in g.edges()}
    # old (0,1) and (1,2) touch the removed region -> gone; old (2,3) -> (1,2)
    assert edges_after == {(1, 2)}, edges_after
    assert all(a < 3 and b < 3 for a, b in edges_after)
    print(f"       edges {edges_before} -> {edges_after}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 指代表 / 自模型跟着走，被删掉的整行丢弃并计数
# ══════════════════════════════════════════════════════════════════════════════

def test_grounding_keys_follow_the_table() -> None:
    t = GroundingTable()
    for _ in range(3):
        t.observe(1, (0, 0))              # this region will be removed
    for _ in range(5):
        t.observe(2, (1, 0))
    for _ in range(4):
        t.observe(3, (2, 0))

    dropped = t.reindex({0: 0, 2: 1, 3: 2})
    assert dropped == 3, dropped
    assert sorted(t.counts) == [1, 2], sorted(t.counts)
    assert t.symbol_totals == {1: 5, 2: 4}, t.symbol_totals
    assert t.total == 9, t.total
    assert t.label(1) == "b1b0" and t.label(2) == "b2b0"
    print(f"       3-observation row dropped, keys now {sorted(t.counts)}")


def test_self_model_keys_follow_the_table() -> None:
    m = OnlineSelfModel()
    for _ in range(4):
        m.observe(1, 2)
        m.observe(2, 3)
    m.reindex({0: 0, 2: 1, 3: 2})
    assert sorted(m._counts) == [1], sorted(m._counts)
    assert m.predict(1) == 2, m.predict(1)
    print(f"       self-model keys now {sorted(m._counts)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 整条链路：她身上不再有悬空编号
# ══════════════════════════════════════════════════════════════════════════════

def test_a_live_stream_leaves_no_dangling_index() -> None:
    from src.common import persist
    from src.L5_cognition.stream import Stream

    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive for this test")
    for _ in range(120):
        st.landscape.set_noise_amplitude(0.02)
        st.step()

    space = st.landscape.state_space
    attrs = space.attractors()
    assert len(attrs) > 20, len(attrs)
    # shrink a region in the middle so the next recover removes it
    victim = len(attrs) // 2
    attrs[victim].amp = 1e-12
    n_before = len(space.attractors())
    dropped_before = int(getattr(st.grounding, "dropped_by_remap", 0))
    for _ in range(40):
        st.landscape.set_noise_amplitude(0.02)
        st.step()

    n = len(space.attractors())
    # The count need not fall: forty more steps also create regions. What must
    # hold is that every stored index still points inside the current table.
    assert all(0 <= s < n for s in st.grounding.counts), sorted(st.grounding.counts)[-3:]
    assert all(0 <= s < n for s in st.self_model._counts), sorted(st.self_model._counts)[-3:]
    assert st._last_symbol is None or 0 <= int(st._last_symbol) < n
    cur = st.detector.current_attractor
    assert cur is None or 0 <= int(cur) < n
    assert all(e.a < n and e.b < n for e in st.graph.edges())
    dropped_after = int(getattr(st.grounding, "dropped_by_remap", 0))
    print(f"       regions {n_before} -> {n}; every stored index < {n}; "
          f"grounding rows dropped by remap: {dropped_after - dropped_before}")


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
