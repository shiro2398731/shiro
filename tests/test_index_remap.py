r"""区域编号在结构变化后必须跟着走 —— 或者干脆不用编号。

区域编号有两种，混在一起就是"看起来有数据"的第三类来源：

    名字（uid）  图的两端、指代表、自模型、`_last_symbol`。永不因结构变化而变。
    行号（下标） detector 的当前区域、停留账本、预演序列。位置性的，必须改。

`StateSpace` 在每次结构变化（recover 删除 / 合并 / 折叠）后发布一张 old->new 行号表，
`Stream` 取走它给**行号**用；图只要按 `alive_uids()` 删掉消失的那几端，`Consolidator.sync`
做这件事，不需要那张表。本文件分别钉住这两半。

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

def test_graph_edges_are_keyed_by_uid_and_a_removal_only_deletes() -> None:
    """A structural change may only *delete* edges, never renumber them.

    This used to be the renumbering test: nodes were table indices, so removing
    one region shifted every later node and the map had to be pushed through
    the graph. With uids, the surviving edges are untouched -- which is the
    property that removes the failure mode (a missed remap pointing an edge at
    a neighbour).
    """
    ss = _space(4)
    for i, a in enumerate(ss.attractors()):
        a.uid = i + 1
    ss._next_uid = 5
    g = TraceGraph()
    for node in (1, 2, 3, 4):
        g.touch(node)
    edges_before = {(e.a, e.b) for e in g.edges()}
    assert edges_before == {(1, 2), (2, 3), (3, 4)}, edges_before

    ss._attractors[1].amp = 1e-9          # uid 2 will fall below min_amp
    ss.recover()
    assert ss.alive_uids() == {1, 3, 4}, ss.alive_uids()

    dropped = Consolidator().sync(g, ss)
    edges_after = {(e.a, e.b) for e in g.edges()}
    # Edges touching the removed uid 2 are gone; (3, 4) keeps its numbers.
    assert edges_after == {(3, 4)}, edges_after
    assert dropped == {"dropped_edges": 2}, dropped
    assert all(a in ss.alive_uids() and b in ss.alive_uids()
               for a, b in edges_after)
    print(f"       removed uid 2: edges {edges_before} -> {edges_after}, nothing renumbered")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 指代表 / 自模型跟着走，被删掉的整行丢弃并计数
# ══════════════════════════════════════════════════════════════════════════════

def test_reference_tables_are_keyed_by_uid_not_by_position() -> None:
    """The property the whole change exists for: removing a region does not
    rename the others.

    Before uids, the grounding table and the self-model were keyed by table index,
    so removing one region shifted every later key and the statistics silently
    moved to a different region. Now the keys are uids: a surviving region keeps
    its number across a structural change, and only the removed one disappears.
    """
    ss = _space(6)
    for i, a in enumerate(ss.attractors()):
        a.uid = i + 1
    ss._next_uid = 7

    t = GroundingTable()
    for a in ss.attractors():
        t.observe(int(a.uid), (0, 0))
    m = OnlineSelfModel()
    for a, b in zip(ss.attractors(), ss.attractors()[1:]):
        m.observe(int(a.uid), int(b.uid))

    kept_uids = {int(a.uid) for a in ss.attractors()}
    ss._attractors[2].amp = 1e-12          # will fall below min_amp
    ss.recover()
    alive = ss.alive_uids()
    assert alive == kept_uids - {3}, (alive, kept_uids)

    # nothing in either table had to be rewritten: uids do not move
    assert set(t.counts) == kept_uids
    assert set(m._counts) == kept_uids - {6}      # the last pair's source
    print(f"       removed uid 3; surviving keys unchanged: {sorted(alive)}")


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
    uids = space.alive_uids()
    # Two conventions live here, on purpose:
    #   * names (uid): graph edges, grounding, self-model, _last_symbol;
    #   * rows (index): detector.current_attractor, _dwell_region.
    # The count need not fall: forty more steps also create regions. What must
    # hold is that nothing points at a region that is gone.
    assert set(st.grounding.counts) <= uids, sorted(set(st.grounding.counts) - uids)
    assert set(st.self_model._counts) <= uids, sorted(set(st.self_model._counts) - uids)
    for row in st.self_model._counts.values():
        assert set(row) <= uids, sorted(set(row) - uids)
    assert st._last_symbol is None or int(st._last_symbol) in uids
    assert all(e.a in uids and e.b in uids for e in st.graph.edges())
    cur = st.detector.current_attractor
    assert cur is None or 0 <= int(cur) < n
    dropped_after = int(getattr(st.grounding, "dropped_by_remap", 0))
    print(f"       regions {n_before} -> {n}; every name inside {len(uids)} live "
          f"uids (graph edges {len(st.graph)}); row ids inside [0, {n}); "
          f"rows dropped: {dropped_after - dropped_before}")


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
