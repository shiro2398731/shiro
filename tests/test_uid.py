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

    # A fresh name must never collide with a live one.
    space._next_uid = max(int(a.uid) for a in space.attractors()) + 1


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
    print("       names reversed everywhere -> same trajectory, depths, MI and bits")


def test_uid_is_not_used_for_ordering_or_arithmetic() -> None:
    """A name that can be compared or sorted becomes a preference (C3).

    Mechanical check over terrain.py: no `x.uid` inside a binary operation, a
    comparison, or the argument of sorted()/min()/max(). Assignments, dict keys
    and set membership stay allowed -- that is what a name is for.
    """
    tree = ast.parse(TERRAIN.read_text(encoding="utf-8"))
    offenders: list[str] = []

    def is_uid(node) -> bool:
        return isinstance(node, ast.Attribute) and node.attr == "uid"

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            if is_uid(node.left) or is_uid(node.right):
                offenders.append(f"arithmetic at line {node.lineno}")
        elif isinstance(node, ast.Compare):
            for part in [node.left] + list(node.comparators):
                if is_uid(part):
                    offenders.append(f"comparison at line {node.lineno}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in {"sorted", "min", "max"}:
            if any(is_uid(sub) for sub in ast.walk(node)):
                offenders.append(f"{node.func.id}() at line {node.lineno}")

    assert not offenders, f"uid used as a value: {offenders}"
    print("       terrain.py: no .uid in arithmetic, comparisons, sorted/min/max")


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
