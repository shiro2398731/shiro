r"""区域编号在结构变化后必须跟着走 —— 或者干脆不用编号。

区域编号有两种，混在一起就是"看起来有数据"的第三类来源：

    名字（uid）  图的两端、指代表、自模型、`_last_symbol`、detector 发布给外部的标签。
                 永不因结构变化而变。
    行号（下标） 并行数组（质心/深度/尺度）里"第几行"。随表变化，但**没有任何地方存它**
                 跨步使用：要用的时候现场查 `uid_row()`。

文件名沿用历史：`_apply_index_remap`、`take_index_remap`、`_publish_remap` 都在 2026-11-14 删掉了
（那张事后补丁表本身就是"漏一次就指错"的来源）。本文件现在钉的是**结论**：
结构变化之后，她身上不许有任何指错的地方。

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
        a = Attractor(mu=np.full(dim, 0.3 * i), amp=0.5)
        a.uid = i + 1
        ss._attractors.append(a)
    ss._next_uid = n + 1
    ss._total_steps = 100
    return ss


# ══════════════════════════════════════════════════════════════════════════════
# 1. 删掉一行，名字不动，行号会动
# ══════════════════════════════════════════════════════════════════════════════

def test_a_removal_moves_rows_but_not_names() -> None:
    """The whole point: the row index shifts, the name does not.

    This used to be a test of the old->new index map. The map is gone; what is
    left to check is the thing the map was papering over -- that no name moves
    when the table does.
    """
    ss = _space(4)
    before = {int(a.uid): i for i, a in enumerate(ss.attractors())}
    assert before == {1: 0, 2: 1, 3: 2, 4: 3}

    ss._attractors[1].amp = 1e-9          # uid 2 will fall below min_amp
    ss.recover()

    after = {int(a.uid): i for i, a in enumerate(ss.attractors())}
    assert set(after) == {1, 3, 4}, after
    assert after[3] != before[3] or after[4] != before[4], "the rows should shift"
    assert ss.uid_row() == after, (ss.uid_row(), after)
    print(f"       removed uid 2: rows {before} -> {after}, names untouched")


def test_a_clean_recover_changes_nothing() -> None:
    ss = _space(4)
    ss.recover()
    assert len(ss.attractors()) == 4
    assert ss.uid_row() == {1: 0, 2: 1, 3: 2, 4: 3}
    print("       nothing removed: 4 rows, 4 names, unchanged")


def test_a_structural_change_does_not_change_her_label() -> None:
    """The id she publishes must keep meaning the same region.

    Remove a region in front of the one she is in, then look: her label is
    unchanged and the row it resolves to has moved up. Before 2026-11-14 the
    label *was* the row, so this removal silently relabelled her -- and the only
    thing standing between that and a wrong broadcast was an old->new map that
    had to reach the detector in time.
    """
    from src.common import persist
    from src.L5_cognition.stream import Stream

    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive for this test")
    for _ in range(60):
        st.landscape.set_noise_amplitude(0.02)
        st.step()

    space = st.landscape.state_space
    label = int(st.detector.current_attractor)
    row_before = space.uid_row()[label]

    victim = None
    for i, a in enumerate(space.attractors()):
        if int(a.uid) != label and i < row_before:
            victim = a
            break
    assert victim is not None, "no region in front of hers to remove"
    victim.amp = 1e-12
    space.recover()

    assert int(st.detector.current_attractor) == label, (
        "her label moved because the table moved",
        st.detector.current_attractor, label)
    assert label in space.alive_uids()
    row_after = space.uid_row()[label]
    assert row_after < row_before, (row_before, row_after)

    st.landscape.set_noise_amplitude(0.02)
    st.step()
    assert int(st.detector.current_attractor) in space.alive_uids()
    print(f"       removed a region in front: label {label} kept, row "
          f"{row_before} -> {row_after}")


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
    for _ in range(40):
        st.landscape.set_noise_amplitude(0.02)
        st.step()

    n = len(space.attractors())
    uids = space.alive_uids()
    # Every id she stores is now a name. The count need not fall: forty more
    # steps also create regions. What must hold is that nothing points at a
    # region that is gone.
    assert set(st.grounding.counts) <= uids, sorted(set(st.grounding.counts) - uids)
    assert set(st.self_model._counts) <= uids, sorted(set(st.self_model._counts) - uids)
    for row in st.self_model._counts.values():
        assert set(row) <= uids, sorted(set(row) - uids)
    assert st._last_symbol is None or int(st._last_symbol) in uids
    assert all(e.a in uids and e.b in uids for e in st.graph.edges())
    assert st._dwell_region is None or int(st._dwell_region) in uids
    cur = st.detector.current_attractor
    assert cur is None or int(cur) in uids, (cur, len(uids))
    assert all(int(u) in uids for u in st.detector.recent_sequence(k=8))
    if st._rehearsal is not None:
        assert all(int(u) in uids or int(u) == -1
                   for u in st._rehearsal["predicted"]), st._rehearsal["predicted"]
    print(f"       regions {n_before} -> {n}; every stored id inside {len(uids)} "
          f"live uids (graph edges {len(st.graph)}, detector label {cur}, "
          f"dwell {st._dwell_region})")


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
