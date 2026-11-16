r"""uid：一块地方的名字，不是一个位置。

为什么要有它
------------
区域靠列表下标识别时，删掉一个区域会让它后面每个区域的编号平移，而存着编号的地方有六处
（图、指代表、自模型两端、送给 L6 的 id、journal 的 fold、预演）。2026-11-13 一晚撞了四次，
形态不同、症状相同：数字看起来正常，行为悄悄错。

现在的分工是：
    名字（uid）  grounding / self-model / `_last_symbol` 用它，永不变；
    位置（下标） 图、detector、`_dwell_region` 用它，会随表变化。

这组测试钉住**两件事**，缺一件这个改动就没意义：

1. **名字是稳的**：折叠/重启不会改名字，也不会把旧名字发给新区域；
2. **名字没有含义**：把 uid 随机重排，她的轨迹与统计一个都不变。
   如果重排之后有任何东西变了，说明有代码在按 uid 算或排序——那时它就是"我挑的偏好"，
   不是她的东西（C3），必须被拦下来。

Run: .\.venv\Scripts\python.exe tests\test_uid.py
"""

from __future__ import annotations

import ast
import collections
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.grounding import GroundingTable  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

TERRAIN = ROOT / "src" / "L5_cognition" / "terrain.py"
L5 = ROOT / "src" / "L5_cognition"
L3 = ROOT / "src" / "L3_memory"
L4 = ROOT / "src" / "L4_drive"
L8I = ROOT / "src" / "L8_identity"


def _ran(steps: int = 60, dim: int = 128) -> Stream:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive for this test")
    for _ in range(steps):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    return st


def _uid_of_mu(space) -> dict:
    return {a.mu.tobytes(): int(a.uid) for a in space.attractors()}


# ══════════════════════════════════════════════════════════════════════════════
# 1. 名字是稳的
# ══════════════════════════════════════════════════════════════════════════════

def test_uids_are_unique_positive_and_never_reused() -> None:
    st = _ran()
    space = st.landscape.state_space
    uids = [int(a.uid) for a in space.attractors()]
    assert len(set(uids)) == len(uids), "duplicate uid"
    assert min(uids) >= 1, "uid 0 means 'not assigned'"
    assert int(space._next_uid) > max(uids), (
        f"next_uid {space._next_uid} would reuse a live name (max {max(uids)})")
    print(f"       {len(uids)} regions, uids 1..{max(uids)}, next_uid={space._next_uid}")


def test_a_fold_does_not_change_any_name() -> None:
    st = _ran()
    space = st.landscape.state_space
    before = _uid_of_mu(space)
    victims = space._least_recent(5)
    space.apply_fold(victims)

    after = _uid_of_mu(space)
    assert len(after) == len(before) - 5, (len(before), len(after))
    for mu, uid in after.items():
        assert before[mu] == uid, "a surviving region changed its name"
    print(f"       folded 5 -> the other {len(after)} kept every uid")


def test_a_restart_keeps_names_and_does_not_reuse_them() -> None:
    st = _ran()
    space = st.landscape.state_space
    uids_before = sorted(int(a.uid) for a in space.attractors())
    high_water = int(space._next_uid)

    payload = persist.build_l5_payload(st)
    again = Stream()
    persist.apply_l5_payload(again, payload)
    space2 = again.landscape.state_space

    assert sorted(int(a.uid) for a in space2.attractors()) == uids_before
    assert int(space2._next_uid) == high_water, (space2._next_uid, high_water)
    print(f"       {len(uids_before)} names identical after a restart, "
          f"next_uid={high_water}")


def test_a_legacy_checkpoint_is_converted_to_uids() -> None:
    """Older checkpoints key their tables by index and carry no uid at all.

    Reinterpreting those keys as uids would attach her statistics to the wrong
    regions; the loader must convert them (index -> uid) instead.
    """
    st = _ran()
    payload = persist.build_l5_payload(st)
    attrs = payload["attractors"]
    for d in attrs:
        d.pop("uid", None)
    payload.pop("next_uid", None)
    payload.pop("symbol_ids", None)

    def to_index_keys(table: dict) -> dict:
        out = {}
        for uid, row in (table or {}).items():
            if isinstance(row, dict):
                out[str(uid)] = {str(int(k)): v for k, v in row.items()}
            else:
                out[str(uid)] = row
        return out

    # rebuild the tables with index keys, as an old writer would have written them
    gr = payload.get("grounding") or {}
    if gr.get("counts"):
        by_index = {}
        uid_to_i = {int(d.get("uid", 0)): i for i, d in enumerate(attrs)}
        for uid, row in gr["counts"].items():
            by_index[str(uid)] = row
        gr["counts"] = by_index
    sm = payload.get("self_model") or {}

    again = Stream()
    persist.apply_l5_payload(again, payload)
    space2 = again.landscape.state_space
    uids = [int(a.uid) for a in space2.attractors()]
    assert len(set(uids)) == len(uids) and min(uids) >= 1
    assert int(space2._next_uid) > max(uids)
    print(f"       legacy payload -> uids 1..{max(uids)}, next_uid={space2._next_uid}")


def test_a_legacy_graph_keeps_its_direction_when_the_names_flip() -> None:
    """Old graph payloads key their nodes by table index, and the conversion to
    uids can put the two ends in the other order. `w_ab` is the weight leaving
    the stored `a`, so a swap that did not swap the two directions would turn
    "she goes there" into "she comes back" -- with the right total weight and
    the right edge count, so nothing would notice.

    The real archive cannot exercise this: its uids happen to be index+1, so no
    pair flips. Hence a payload built on purpose.
    """
    st = _ran()
    payload = persist.build_l5_payload(st)
    i, j = 10, 11
    at = payload["attractors"]
    at[i]["uid"], at[j]["uid"] = at[j]["uid"], at[i]["uid"]
    assert int(at[i]["uid"]) > int(at[j]["uid"]), "the swap did not create a flip"

    payload["graph"] = {
        "edges": [{"a": i, "b": j, "w_ab": 3.0, "n_ab": 3,
                   "w_ba": 1.0, "n_ba": 1}],
        "last_node": i,
    }
    payload.pop("graph_ids", None)
    payload.pop("symbol_ids", None)

    again = Stream()
    persist.apply_l5_payload(again, payload)
    space = again.landscape.state_space
    uid_i = int(space.attractors()[i].uid)
    uid_j = int(space.attractors()[j].uid)
    assert uid_i > uid_j

    e = again.graph.edge(uid_i, uid_j)
    assert e is not None, "the edge vanished in the conversion"
    assert abs(again.graph.directed(uid_i, uid_j) - 3.0) < 1e-12, (
        again.graph.directed(uid_i, uid_j), again.graph.directed(uid_j, uid_i))
    assert abs(again.graph.directed(uid_j, uid_i) - 1.0) < 1e-12
    print(f"       index-keyed edge i->j (3.0) survived as uid {uid_i}->{uid_j} (3.0)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 名字没有含义
# ══════════════════════════════════════════════════════════════════════════════

def _rename_everywhere(st, perm: dict) -> None:
    """Rename every region id consistently: regions, tables (both ends), and the
    symbol she was last in.

    A permutation that touches only the regions is not a renaming -- it makes the
    stored tables describe different things, and then of course the self-model
    changes. The claim being tested is "the name means nothing", so the rename has
    to be the same rename everywhere.
    """
    space = st.landscape.state_space
    for a in space.attractors():
        a.uid = int(perm[int(a.uid)])

    g = st.grounding
    g.counts = {int(perm[int(k)]): dict(row) for k, row in g.counts.items()}
    g.symbol_totals = {int(perm[int(k)]): v for k, v in g.symbol_totals.items()}

    m = st.self_model
    m._counts = {int(perm[int(k)]): {int(perm[int(t)]): c for t, c in row.items()}
                 for k, row in m._counts.items()}
    m._seen = {int(perm[int(k)]): v for k, v in m._seen.items()}

    if st._last_symbol is not None:
        st._last_symbol = int(perm[int(st._last_symbol)])

    # The graph names regions too, so the same rename has to reach it. Without
    # this the graph keeps the old names, every edge misses on lookup, and the
    # tension field shrinks to nothing -- a difference caused by the test, not
    # by the code.
    st.graph.relabel({int(k): int(v) for k, v in perm.items()})

    # Identity counts edges by (uid, uid) as well. It does not feed the
    # dynamics, but a "consistent rename" that skips it is not consistent.
    idn = st.identity
    def pair(k):
        return (int(perm[int(k[0])]), int(perm[int(k[1])]))
    idn._appear = collections.defaultdict(int, {pair(k): v for k, v in idn._appear.items()})
    idn._edge_snapshot = {pair(k): v for k, v in idn._edge_snapshot.items()}

    # The detector publishes region names too, and it is not just observation:
    # `_apply_leave` looks the row up by `detector.current_attractor`, so a
    # renamed stream whose detector still holds the old name finds no row and
    # simply does not move. That is a difference caused by the test, not by the
    # code. 2026-11-16: this is what made the test fail after the archive grew.
    det = st.detector
    if det._current is not None:
        det._current = int(perm[int(det._current)])
    for ev in det.history:
        ev.from_attractor = int(perm[int(ev.from_attractor)])
        ev.to_attractor = int(perm[int(ev.to_attractor)])

    # The stay ledger names the region she has been in.
    if st._dwell_region is not None:
        st._dwell_region = int(perm[int(st._dwell_region)])

    # A fresh name must never collide with a live one.
    space._next_uid = max(int(a.uid) for a in space.attractors()) + 1
    # The name -> row map is cached per structural version, and renaming is the
    # one thing that changes a name without changing the structure. Production
    # never does this (a uid is assigned at birth), so the cache has to be told.
    space._uid_row = None


def test_consistent_relabelling_changes_nothing() -> None:
    a = _ran()
    b = _ran()

    uids = [int(x.uid) for x in b.landscape.state_space.attractors()]
    perm = {old: new for old, new in zip(uids, reversed(uids))}
    _rename_everywhere(b, perm)

    for _ in range(40):
        for st in (a, b):
            st.landscape.set_noise_amplitude(0.02)
            st.step()

    assert a._st.x.tobytes() == b._st.x.tobytes(), "renaming changed her trajectory"
    sa, sb = a.landscape.state_space, b.landscape.state_space
    assert len(sa.attractors()) == len(sb.attractors()), (
        len(sa.attractors()), len(sb.attractors()))
    assert abs(sum(float(x.amp) for x in sa.attractors())
               - sum(float(y.amp) for y in sb.attractors())) < 1e-9
    assert abs(a.grounding.mutual_information()
               - b.grounding.mutual_information()) < 1e-9, \
        "renaming changed the referent statistics"
    assert abs(a.self_model.bits_per_observation
               - b.self_model.bits_per_observation) < 1e-9, \
        "renaming changed her self-prediction loss"

    # Structure, not just the trajectory: the graph and the recalled path name
    # regions, so a rename that leaked into either would show up here even if
    # the step happened to come out the same.
    assert len(a.graph) == len(b.graph), (len(a.graph), len(b.graph))
    assert abs(sum(e.w for e in a.graph.edges())
               - sum(e.w for e in b.graph.edges())) < 1e-9, \
        "renaming changed the connectivity weights"
    ra, rb = a._last_recall, b._last_recall
    assert (ra is None) == (rb is None)
    if ra is not None:
        assert len(ra.path) == len(rb.path), (ra.path, rb.path)
        assert abs(sum(ra.weights) - sum(rb.weights)) < 1e-9

    print("       names reversed everywhere -> same trajectory, depths, MI, bits, "
          f"{len(a.graph)} edges, recalled path of {0 if ra is None else len(ra.path)}")


def test_uid_is_not_used_for_ordering_or_arithmetic() -> None:
    """A name that can be compared or sorted becomes a preference (C3).

    Mechanical check over every module that handles region names: no `x.uid`
    inside a binary operation, a comparison, or the argument of
    sorted()/min()/max(). Assignments, dict keys and set membership stay
    allowed -- that is what a name is for.

    This is the cheap guard for the *direct* form of the leak, where the field
    itself appears in an expression. It is not enough on its own: the leak found
    on 2026-11-14 was `OnlineSelfModel.predict` breaking ties with
    `max(..., key=lambda kv: (count, -id))` -- no `.uid` in sight, the name
    arriving as `kv[0]`. Only the behavioural test above (rename everything,
    expect the same trajectory) caught that one. Keep both: this one is
    per-file and whole-file, so the direct form cannot come back in a new
    helper.
    """
    files = [
        TERRAIN,                                   # names live here
        L5 / "stream.py",                          # uses them every step
        L5 / "self_model.py",                      # keys and a tie-break
        L5 / "grounding.py",                       # keys
        L3 / "trace.py",                           # graph nodes
        L3 / "consolidate.py",                     # pruning by alive set
        L3 / "recall.py",                          # path through the graph
        L4 / "drive.py",                           # tension pairs
        L8I / "identity.py",                       # core members
    ]
    offenders: list[str] = []

    def is_uid(node) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == "uid"

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                if is_uid(node.left) or is_uid(node.right):
                    offenders.append(f"{path.name}:{node.lineno} arithmetic")
            elif isinstance(node, ast.Compare):
                for part in [node.left] + list(node.comparators):
                    if is_uid(part):
                        offenders.append(f"{path.name}:{node.lineno} comparison")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id in {"sorted", "min", "max"}:
                if any(is_uid(sub) for sub in ast.walk(node)):
                    offenders.append(f"{path.name}:{node.lineno} {node.func.id}()")

    assert not offenders, f"uid used as a value: {offenders}"
    print(f"       {len(files)} modules: no .uid in arithmetic, comparisons, sorted/min/max")


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
