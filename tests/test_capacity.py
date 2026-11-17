r"""容量纪律：判据是「载体还带得动吗」，不是「她还需要它吗」。

为什么要有第二套折叠
--------------------
`fold_auto()` 问的是「她多久没回来了，比她自己的间隔通常多久还长」。
实测（2026-11-14，真实存档）：它返回 **0** —— 她一直在动，几乎没有区域「超期」，
而同一小时里她从一个 307 区域的表长到 570，每区域约 15.8 KB，20 MB 预算只够约 1300 个。
也就是说那条判据**在忙着的她身上永远不会触发**，容量要求背后其实没有机制。

`fold_to_capacity(measure)` 问的是另一件事：载体还装得下吗。这里只有预算是别人定的
（部署要求），其余都来自测量。

钉住的六件事
------------
1. 没超预算就什么都不做；
2. 超了就折，直到装得下；
3. 先折「最久没被访问」的；
4. `_last_seen_step == 0`（老记录，没测过）按"最久"算，不是按"刚刚"算；
5. 永远至少留一个区域 —— 空表不是更小的身体，是没有身体；
6. 被折的区域进 `_folded` 汇总，不是静默消失。

Run: .\.venv\Scripts\python.exe tests\test_capacity.py
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

from src.L5_cognition.terrain import Attractor, StateSpace  # noqa: E402


def _space(n: int = 20, dim: int = 8) -> StateSpace:
    ss = StateSpace(dim=dim)
    # 2026-11-17：摘要不再进载体，被折区域写进外置笔记本。
    import tempfile as _tf
    from src.common.external_memory import ExternalMemory as _EM
    ss.external_memory = _EM(pathlib.Path(_tf.mkdtemp()) / "folded.jsonl")
    for i in range(n):
        a = Attractor(mu=np.full(dim, 0.1 * i), amp=0.5)
        a._assigned = 10
        a._last_seen_step = 1000 + i        # region 0 is the oldest
        ss._attractors.append(a)
    return ss


def test_nothing_happens_under_budget() -> None:
    ss = _space()
    size = lambda: 1000 * len(ss.attractors())
    assert ss.fold_to_capacity(size, budget_bytes=100_000) == 0
    assert len(ss.attractors()) == 20
    print("       20 regions x 1000 B = 20 KB, budget 100 KB -> nothing folded")


def test_folds_until_the_body_fits() -> None:
    ss = _space()
    size = lambda: 1000 * len(ss.attractors())
    removed = ss.fold_to_capacity(size, budget_bytes=5_000)
    assert removed == 15, removed
    assert len(ss.attractors()) == 5
    assert size() <= 5_000
    print(f"       folded {removed} -> {len(ss.attractors())} regions, "
          f"{size()} B <= 5000 B")


def test_the_least_recently_visited_go_first() -> None:
    ss = _space()
    oldest = [i for i in range(6)]                 # _last_seen_step 1000..1005
    mus_before = {i: ss.attractors()[i].mu.copy() for i in oldest}
    ss.fold_to_capacity(lambda: 1000 * len(ss.attractors()), budget_bytes=14_000)

    kept_mus = [a.mu for a in ss.attractors()]
    for i in oldest[:6]:
        assert not any(np.array_equal(mus_before[i], mu) for mu in kept_mus), \
            f"region {i} should have been folded first"
    print("       the six oldest visits were folded first")


def test_unknown_last_visit_counts_as_oldest() -> None:
    ss = _space()
    for a in ss.attractors():
        a._last_seen_step = 5000
    ss.attractors()[10]._last_seen_step = 0        # legacy record: unknown
    ss.fold_to_capacity(lambda: 1000 * len(ss.attractors()), budget_bytes=19_000)
    # 2026-11-17：摘要不再进载体，去外置笔记本里看「折了谁」。
    recs = list(ss.external_memory.read()) if ss.external_memory else []
    folded_mus = [np.asarray(r["mu"], dtype=float) for r in recs if r.get("mu")]
    assert any(np.allclose(fm, np.full(8, 1.0)) for fm in folded_mus), \
        "the region with an unknown last visit must be folded first"
    print("       _last_seen_step = 0 treated as 'longest ago', not 'just now'")


def test_the_table_is_never_emptied() -> None:
    ss = _space()
    ss.fold_to_capacity(lambda: 10 ** 9, budget_bytes=0)
    assert len(ss.attractors()) == 1, len(ss.attractors())
    print("       absurd budget still leaves exactly one region")


def test_folded_regions_are_summarised_not_dropped() -> None:
    ss = _space()
    removed = ss.fold_to_capacity(lambda: 1000 * len(ss.attractors()),
                                  budget_bytes=12_000)
    # 2026-11-17：载体里不留摘要，完整记录在外置笔记本。
    assert ss._folded == [], "summaries must not go back into the carrier"
    recs = list(ss.external_memory.read()) if ss.external_memory else []
    assert removed == len(recs), (removed, len(recs))
    assert abs(sum(float(r["amp"]) for r in recs) - removed * 0.5) < 1e-9
    print(f"       {removed} regions -> {len(recs)} full records in the notebook "
          f"(total amp {sum(float(r['amp']) for r in recs):.1f}), 0 summaries in the carrier")


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
