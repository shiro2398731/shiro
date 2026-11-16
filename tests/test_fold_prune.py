r"""折叠之后，指代表与自模型必须跟着剪枝。

为什么有这组测试
----------------
容量折叠（`fold_to_capacity`）把区域的 uid 从 `alive_uids()` 里拿走。但两张按
uid 记账的表没有跟着剪：

- `GroundingTable.counts` / `.symbol_totals`  —— 符号是 uid
- `OnlineSelfModel._counts` / `._seen`        —— 两端都是 uid

2026-11-15 14:55 起折叠第一次真触发，到 2026-11-16 00:35 实测：
区域 1074，指代键 3772，自模型符号 6326，`tools/integrity.py` 报 **33772 条**
`grounding.symbol=X is not a live uid`。

这组测试钉住三件事：
1. 死 uid 被删，活的一个不动；
2. 删完之后 integrity 会查的那几条恒等式仍然成立（total / symbol_totals /
   context_totals 三处对得上；`_counts` 两端都能在 `_seen` 里找到）；
3. 累计量（nll_bits / hits / observations …）不因为剪枝被改写——那是历史。

Run: .\.venv\Scripts\python.exe tests\test_fold_prune.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.grounding import GroundingTable  # noqa: E402
from src.L5_cognition.self_model import OnlineSelfModel  # noqa: E402


# ---------- grounding ----------

def test_grounding_prune_drops_only_dead_symbols() -> None:
    g = GroundingTable()
    g.observe(10, (1, 1))
    g.observe(10, (2, 3))
    g.observe(11, (1, 1))
    g.observe(12, (0, 0))
    assert g.total == 4, g.total

    dropped = g.prune({10, 12})
    assert dropped == 1, dropped
    assert set(g.counts) == {10, 12}, sorted(g.counts)
    assert g.total == 3, g.total
    print(f"       dropped 1 dead symbol, 2 live kept, total 4 -> {g.total}")


def test_grounding_prune_keeps_totals_consistent() -> None:
    """integrity 第三条：counts 加得起来，三本账互相一致。"""
    g = GroundingTable()
    for sym in (1, 2, 3, 4):
        for ctx in ((0, 0), (1, 2)):
            g.observe(sym, ctx)

    g.prune({2, 4})

    total = 0
    sym_totals: dict[int, int] = {}
    ctx_totals: dict = {}
    for s, row in g.counts.items():
        for c, k in row.items():
            total += int(k)
            sym_totals[s] = sym_totals.get(s, 0) + int(k)
            ctx_totals[c] = ctx_totals.get(c, 0) + int(k)

    assert total == int(g.total), (total, g.total)
    assert sym_totals == {int(k): int(v) for k, v in g.symbol_totals.items()}
    assert ctx_totals == {k: int(v) for k, v in g.context_totals.items()}
    print(f"       after prune: total={g.total}, symbols={len(sym_totals)}, contexts={len(ctx_totals)}")


def test_grounding_prune_is_a_noop_when_everything_is_alive() -> None:
    g = GroundingTable()
    g.observe(5, (1, 1))
    g.observe(6, (1, 1))
    before = (dict(g.counts), dict(g.symbol_totals), dict(g.context_totals), g.total)

    assert g.prune({5, 6}) == 0
    assert (dict(g.counts), dict(g.symbol_totals), dict(g.context_totals), g.total) == before
    print("       all alive -> 0 dropped, table untouched")


# ---------- self model ----------

def test_self_model_prune_drops_dead_ends() -> None:
    m = OnlineSelfModel()
    for a, b in ((7, 8), (8, 9), (9, 7), (7, 9)):
        m.observe(a, b)

    assert m.prune({7, 8, 9}) == 0, "nothing is dead"
    dropped = m.prune({7, 8})

    assert 9 not in m._seen, sorted(m._seen)
    for prev, row in m._counts.items():
        assert 9 not in row, (prev, row)
    assert dropped >= 1, dropped
    print(f"       dropped {dropped} row(s) touching uid 9; _seen now {sorted(m._seen)}")


def test_self_model_prune_keeps_seen_consistent() -> None:
    """integrity 第二次检查：每个符号与每个目标都必须出现在 `_seen` 里。"""
    m = OnlineSelfModel()
    for a, b in ((1, 2), (2, 3), (3, 1), (1, 4), (4, 1)):
        m.observe(a, b)

    m.prune({1, 2, 3})

    seen = {int(k) for k in m._seen}
    for s, row in m._counts.items():
        assert int(s) in seen, f"symbol {s} missing from _seen"
        for t in row:
            assert int(t) in seen, f"target {t} missing from _seen"
    print(f"       after prune: seen={sorted(seen)}, rows={len(m._counts)}")


def test_prune_does_not_rewrite_history() -> None:
    """累计量是历史。剪枝不许碰它——那比留着一个死 id 更坏。"""
    m = OnlineSelfModel()
    for a, b in ((1, 2), (2, 3), (3, 1)):
        m.observe(a, b)

    before = (m.observations, m.hits, m.nll_bits, m.scored, m.conf_sum, m.conf_n)
    m.prune({1, 2})
    after = (m.observations, m.hits, m.nll_bits, m.scored, m.conf_sum, m.conf_n)

    assert before == after, (before, after)
    print(f"       history intact: observations={m.observations}, nll_bits={m.nll_bits:.4f}")


def test_prune_treats_uids_as_names_not_quantities() -> None:
    """同一套结构、换一套名字，剪枝结果必须一样（名字没有含义）。"""
    a = GroundingTable()
    b = GroundingTable()
    for sym in (100, 200, 300):
        a.observe(sym, (1, 1))
        b.observe(sym * 7, (1, 1))

    da = a.prune({100, 300})
    db = b.prune({700, 2100})

    assert da == db == 1, (da, db)
    assert len(a.counts) == len(b.counts) == 2
    print("       renaming the uids changed nothing")


def test_identity_prune_drops_dead_endpoints() -> None:
    """`_appear` 的键是边的两端 uid，任一端死了这条统计就该走。"""
    from src.L8_identity.identity import Identity  # noqa: PLC0415

    ident = Identity()
    ident._appear[(1, 2)] = 3      # 两端都活
    ident._appear[(2, 3)] = 1      # 3 死了
    ident._appear[(3, 1)] = 2      # 3 死了

    dropped = ident.prune({1, 2})
    assert dropped == 2, dropped
    assert set(ident._appear) == {(1, 2)}, dict(ident._appear)
    print(f"       dropped {dropped} edge(s); surviving: {sorted(ident._appear)}")


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
