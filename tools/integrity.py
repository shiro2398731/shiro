"""
tools/integrity.py — 她身上的一致性检查（只读）。

为什么需要这个
--------------
今晚（2026-11-13/14）同一族错误出现了四次：**某个地方存着一个区域编号，而表的形状变了**。
每次的形态不同（错编号、旧缓存、越界索引、悬空键），但症状相同：数字看起来正常，行为悄悄错。

这类错误不会抛异常，只会让她的统计慢慢失真。所以要有一个地方能一次问清楚：
"她现在身上，还有没有指错的地方？"

检查项
------
1. 编号范围：图的边与 `_last_node`、指代表的符号、自模型的键（**两端**）、
   `_last_symbol`、`_dwell_region`、身份表的边——全部必须 < 区域数。
2. 数值：质心有限且量级 < 1e6；深度有限、非负、不超过 ceiling。
3. 计数一致：指代表的 `symbol_totals`/`context_totals`/`total` 必须能由 counts 重算出来；
   自模型同理。
4. 折叠汇总：每条 summary 的质心维度正确、count 为正。
5. 日志：能解析；若有不可用记录，`replay()` 会整本作废（这里只报告数量）。

只读：不修改任何东西，所以可以在她运行的时候跑。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import journal as journal_mod  # noqa: E402
from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

SAVE_LIMIT = 1e6


def check(stream, journal_path: str | None = "data/journal/l5.wal") -> list[str]:
    """Return a list of problems. Empty means consistent."""
    problems: list[str] = []
    space = stream.landscape.state_space
    attrs = space.attractors()
    n = len(attrs)
    if n == 0:
        return ["no regions at all"]

    def _idx(name: str, value) -> None:
        try:
            v = int(value)
        except (TypeError, ValueError):
            problems.append(f"{name}: not an integer ({value!r})")
            return
        if not (0 <= v < n):
            problems.append(f"{name}={v} outside [0, {n})")

    # 1. every stored region id -- and there are two kinds now
    #
    #   uids     grounding / self-model / _last_symbol.  A name, stable across
    #            merges and restarts. Legal values are the live uids, which are
    #            NOT bounded by the region count (uid 400 of 300 regions is fine,
    #            that just means 100 regions were folded before it).
    #   indices  graph edges, detector.current_attractor, _dwell_region.
    #            Positions: legal values are [0, n).
    #
    # Checking a uid against the count (or an index against the uid set) would
    # report nonsense; keeping the two apart is the whole point of the change.
    alive = space.alive_uids()

    def _uid(name: str, value) -> None:
        try:
            v = int(value)
        except (TypeError, ValueError):
            problems.append(f"{name}: not an integer ({value!r})")
            return
        if v not in alive:
            problems.append(f"{name}={v} is not a live uid")

    for e in stream.graph.edges():
        _idx("graph.edge.a", e.a)
        _idx("graph.edge.b", e.b)
    last_node = getattr(stream.graph, "_last_node", None)
    if last_node is not None:
        _idx("graph._last_node", last_node)

    for s in stream.grounding.counts:
        _uid("grounding.symbol", s)
    for s in stream.grounding.symbol_totals:
        _uid("grounding.symbol_totals", s)
    for s, row in stream.self_model._counts.items():
        _uid("self_model.symbol", s)
        for t in row:
            _uid("self_model.target", t)
    if stream._last_symbol is not None:
        _uid("stream._last_symbol", stream._last_symbol)
    if getattr(stream, "_dwell_region", None) is not None:
        _idx("stream._dwell_region", stream._dwell_region)
    for (a, b) in getattr(stream.identity, "_appear", {}):
        _idx("identity.appear.a", a)
        _idx("identity.appear.b", b)

    # Every region must carry a distinct, positive name.
    uids = [int(a.uid) for a in attrs]
    if len(set(uids)) != len(uids):
        problems.append("two regions share a uid")
    if any(u <= 0 for u in uids):
        problems.append("a region has no uid (0)")
    if int(getattr(space, "_next_uid", 1)) <= max(uids or [0]):
        problems.append("next_uid does not exceed the largest live uid (a name could be reused)")

    # 2. numbers
    for i, a in enumerate(attrs):
        mu = np.asarray(a.mu, dtype=float)
        if mu.shape != (space.D,):
            problems.append(f"region {i}: mu shape {mu.shape} != ({space.D},)")
        if not np.isfinite(mu).all():
            problems.append(f"region {i}: centroid not finite")
        elif float(np.abs(mu).max()) > SAVE_LIMIT:
            problems.append(f"region {i}: |centroid| {float(np.abs(mu).max()):.3e} > {SAVE_LIMIT:.0e}")
        amp = float(a.amp)
        if not np.isfinite(amp) or amp < 0:
            problems.append(f"region {i}: depth {amp!r}")
        elif amp > float(space.ceiling) + 1e-9:
            problems.append(f"region {i}: depth {amp:.4f} above ceiling {space.ceiling}")

    # 3. counts add up
    g = stream.grounding
    total = 0
    sym: dict[int, int] = {}
    ctx: dict = {}
    for s, row in g.counts.items():
        for c, k in row.items():
            total += int(k)
            sym[s] = sym.get(s, 0) + int(k)
            ctx[c] = ctx.get(c, 0) + int(k)
    if total != int(g.total):
        problems.append(f"grounding.total {g.total} != sum(counts) {total}")
    if sym != {int(k): int(v) for k, v in g.symbol_totals.items()}:
        problems.append("grounding.symbol_totals disagrees with counts")
    if ctx != {k: int(v) for k, v in g.context_totals.items()}:
        problems.append("grounding.context_totals disagrees with counts")

    m = stream.self_model
    # `observe()` records both ends in `_seen`, so every symbol and every target
    # it counts must appear there. A target that does not is a dangling id.
    for s, row in m._counts.items():
        if int(s) not in {int(k) for k in m._seen}:
            problems.append(f"self_model symbol {s} missing from _seen")
        for t, c in row.items():
            if float(c) <= 0:
                problems.append(f"self_model count {s}->{t} is {c!r}")
            if int(t) not in {int(k) for k in m._seen}:
                problems.append(f"self_model target {t} missing from _seen")

    # 4. folded summaries
    for i, f in enumerate(getattr(space, "_folded", []) or []):
        mu = np.asarray(f.get("mu", []), dtype=float)
        if mu.shape != (space.D,):
            problems.append(f"folded[{i}]: mu dim {mu.size} != {space.D}")
        if int(f.get("count", 0)) <= 0:
            problems.append(f"folded[{i}]: count {f.get('count')!r} not positive")

    # 5. journal
    if journal_path:
        jp = Path(journal_path)
        if jp.exists():
            try:
                j = journal_mod.Journal(jp, dim=space.D)
                bad = j._scan_bad()
                if bad:
                    problems.append(f"journal: {bad} unusable record(s); replay would abort")
            except Exception as e:  # noqa: BLE001
                problems.append(f"journal: unreadable ({type(e).__name__}: {e})")

    return problems


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="consistency check on her state (read-only)")
    ap.add_argument("--journal", type=str, default="data/journal/l5.wal")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    st = Stream()
    if not persist.load_l5(st):
        print("no archive: run P2 first, or point state_dir at one", file=sys.stderr)
        return 2

    problems = check(st, args.journal)
    space = st.landscape.state_space
    if args.json:
        print(json.dumps({"regions": len(space.attractors()),
                          "edges": len(st.graph),
                          "problems": problems}, ensure_ascii=False))
        return 1 if problems else 0

    print(f"regions={len(space.attractors())}  edges={len(st.graph)}  "
          f"grounding_keys={len(st.grounding.counts)}  "
          f"self_model_symbols={len(st.self_model._counts)}")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems[:40]:
            print("  -", p)
        return 1
    print("consistent: every stored region id is in range, counts add up, no poisoned record")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
