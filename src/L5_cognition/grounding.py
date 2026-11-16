"""
src/L5_cognition/grounding.py

符号 ↔ 指代：她的区域编号到底指着什么。

不是什么
--------
不宣称"语义"。这里没有词、没有概念、没有解释。它只做一件可测的事：
**统计一个符号出现时，她自己内部的量处在什么位置**，然后把那个位置变成一个离散标签。

为什么先做这一步
----------------
L6 现在拿到的符号只有 `attractor_id` —— 一个编号。编号本身不指代任何东西
（"12 号区域"对谁都一样）。若要让"符号 → 语句"有意义，符号必须先和她自己的状态
挂钩，否则翻译器只能照着编号造句，那仍然是人在说话（C3）。

可测判据（这是 M1 的思路，用在同一件事上）
----------------------------------------
    I(符号; 指代)  经验互信息，单位 bits
    打乱指代序列后重算的 I  作为零假设的下界（经验 MI 有有限样本偏置）

实测（她的真实存档，4000 步；`python tools/grounding.py`）：
    plug-in MI(符号; 状态桶) = 0.6096 bits，打乱后 0.0324 bits（18.8x）
    **但留出法增益 = -0.0517 bits** —— 表记住了样本，符号对瞬时状态量没有指代。
    留出法另测 p(下一个区域 | 当前区域)：3.3325 → 0.7226 bits，增益 +2.6099（78.3%）；
    换成复合符号 (上一个, 当前) 后为 +2.5118（75.4%）——**组合没有增加带宽**，
    因为对"下一步"来说 (前,现) 在给定"现"之后没有额外信息（一阶链的性质）。

所以本模块目前能支持的说法只有一句：**符号不贴着某个内在标量，但带着关于她下一步的信息**。
plug-in MI 在样本稀疏时严重上偏，判据必须是留出法（`tools/grounding.py` 的第二、三项）；
本文件的 `mutual_information()` 只能当描述性统计，不能单独作为"指代成立"的证据。

为什么指代必须是离散的
----------------------
ARCHITECTURE §4.2：L6 不能拿到任何向量。所以指代以**标签**形式交出
（`label()` 返回如 "b2b1"，或 "none"）。L6 得到的仍然全是符号，
只是这些符号现在指着一件可测的事。

参数分级（docs §11）
--------------------
BUCKETS / HIST / MIN_HIST 都属 C1：一个是分辨率、两个是缓冲区。没有一个是判据。
判据是"互信息是否显著高于打乱后的零假设"，两者都由她自己的数据算出。
"""

from __future__ import annotations

import collections
import math
from typing import Optional

import numpy as np

# Resolution of the referent space: how many quantile buckets each measured
# quantity is cut into. C1. Four buckets = "low / below middle / above middle /
# high", measured against her own history, never against a fixed range.
BUCKETS = 4

# History kept per measured quantity, used to place the current value.
# C1 (buffer length), not a criterion.
HIST = 512

# Fewest samples before her own quantiles mean anything. C1.
MIN_HIST = 64

# Laplace smoothing for the held-out criterion. C1: it keeps an unseen cell
# from producing an infinite loss; it does not change what is being measured.
ALPHA = 0.5


def heldout_gain(pairs: list, alpha: float = ALPHA) -> dict:
    """Held-out log-loss of p(target | symbol) vs p(target): the real criterion.

    `pairs` is [(symbol, target), ...] in time order; the first half fits, the
    second half scores. Gain is bits per observation; positive means knowing
    the symbol helps predict the target on data the table never saw. `symbol`
    may be any hashable value (an id, or a tuple for a composite symbol).

    Why this and not mutual_information()
    -------------------------------------
    Plug-in MI rises with the number of cells even when the variables are
    independent, and it rises fast once cells outnumber samples. Measured on
    her archive: I(region; speed/tension buckets) = 0.6096 bits against a
    shuffled null of 0.0324 (18.8x), which reads like a result. The same data
    scored held-out gives -0.0517 bits, i.e. worse than the marginal: the table
    memorised the sample. A statistic that cannot tell those two apart is not a
    criterion, so this function is what the reports quote.
    """
    split = len(pairs) // 2
    train, test = pairs[:split], pairs[split:]
    if not train or not test:
        return {"n_test": 0}

    targets = sorted({y for _, y in train})
    base = {}
    for _, y in train:
        base[y] = base.get(y, 0) + 1
    n_train = sum(base.values())
    p_base = {y: (base[y] + alpha) / (n_train + alpha * len(targets))
              for y in targets}

    row = {}
    tot = {}
    for s, y in train:
        if s is None:
            continue
        r = row.setdefault(s, {})
        r[y] = r.get(y, 0) + 1
        tot[s] = tot.get(s, 0) + 1

    ll_base = 0.0
    ll_model = 0.0
    n = 0
    for s, y in test:
        if y not in p_base:
            continue
        p = p_base[y]
        if s is not None and tot.get(s):
            p = (row[s].get(y, 0) + alpha) / (tot[s] + alpha * len(targets))
        ll_base += -math.log2(p_base[y])
        ll_model += -math.log2(p)
        n += 1
    if n == 0:
        return {"n_test": 0}
    return {
        "n_test": n,
        "symbols_used": len(tot),
        "distinct_targets": len(targets),
        "baseline_bits": round(ll_base / n, 4),
        "model_bits": round(ll_model / n, 4),
        "gain_bits": round((ll_base - ll_model) / n, 4),
        "gain_share": round((ll_base - ll_model) / ll_base, 4) if ll_base else None,
    }


class GroundingTable:
    """Empirical symbol -> referent statistics. Holds numbers, produces labels."""

    def __init__(self, buckets: int = BUCKETS, hist: int = HIST) -> None:
        self.buckets = int(buckets)
        self.hist = int(hist)
        self._hist: dict[str, collections.deque] = {}
        # counts[symbol][context] = n
        self.counts: dict[int, dict[tuple[int, ...], int]] = {}
        self.symbol_totals: dict[int, int] = {}
        self.context_totals: dict[tuple[int, ...], int] = {}
        self.total = 0

    # ---------- measuring a quantity against her own history ----------

    def bucket(self, name: str, value: float) -> Optional[int]:
        """Where `value` falls in her own history of `name` (0..buckets-1).

        Returns None while the history is too short to place anything: an
        unmeasured value is not bucketed into "low" by default.
        """
        h = self._hist.setdefault(name, collections.deque(maxlen=self.hist))
        v = float(value)
        out: Optional[int] = None
        if len(h) >= MIN_HIST and math.isfinite(v):
            arr = np.asarray(h, dtype=float)
            edges = [float(np.quantile(arr, i / self.buckets))
                     for i in range(1, self.buckets)]
            out = int(np.searchsorted(edges, v))
        h.append(v)
        return out

    # ---------- observation ----------

    def observe(self, symbol: int, context: tuple[int, ...]) -> None:
        """One observation: this symbol occurred while she was in this context."""
        self._add(int(symbol), tuple(int(x) for x in context), 1)

    def _add(self, symbol: int, context: tuple[int, ...], count: int) -> None:
        """Add `count` observations at once (also used when restoring a table)."""
        row = self.counts.setdefault(symbol, {})
        row[context] = row.get(context, 0) + int(count)
        self.symbol_totals[symbol] = self.symbol_totals.get(symbol, 0) + int(count)
        self.context_totals[context] = self.context_totals.get(context, 0) + int(count)
        self.total += int(count)

    # ---------- folding ----------

    def prune(self, alive: set[int]) -> int:
        """删掉已经不存在的区域的统计，返回被删的符号行数。

        折叠会把区域的 uid 从 `alive_uids()` 里拿走，但这张表是按 uid 存的，
        于是它还在给一个她已经没有了的指代记账。2026-11-15 容量折叠第一次真触发
        （区域 1074，表里 3772 个键），integrity 报 33772 条 `not a live uid`。

        键是名字不是量：这里只做集合成员判断，不排序、不比较大小（见 tests/test_uid.py）。

        删完之后 `total` / `symbol_totals` / `context_totals` 三处必须同时减，
        否则 integrity 的第三条（counts 加得起来）会红。
        """
        alive = {int(a) for a in alive}
        dead = [s for s in list(self.counts) if int(s) not in alive]
        if not dead:
            return 0
        for s in dead:
            row = self.counts.pop(s, {})
            self.symbol_totals.pop(s, None)
            for c, k in row.items():
                left = int(self.context_totals.get(c, 0)) - int(k)
                if left > 0:
                    self.context_totals[c] = left
                else:
                    self.context_totals.pop(c, None)
                self.total -= int(k)
        return len(dead)

    # ---------- statistics ----------

    def mutual_information(self) -> float:
        """I(symbol; context) in bits, from the table as it stands."""
        n = self.total
        if n <= 0:
            return 0.0
        total = 0.0
        for s, row in self.counts.items():
            ps = self.symbol_totals[s] / float(n)
            for c, k in row.items():
                pj = k / float(n)
                pc = self.context_totals[c] / float(n)
                if pj > 0.0 and pc > 0.0 and ps > 0.0:
                    total += pj * math.log2(pj / (ps * pc))
        return max(total, 0.0)

    def _entropy_context(self) -> float:
        return self.entropy_context()

    def entropy_context(self) -> float:
        """H(context) in bits: how much her internal state varies, as bucketed."""
        n = self.total
        if n <= 0:
            return 0.0
        return -sum((k / float(n)) * math.log2(k / float(n))
                    for k in self.context_totals.values() if k > 0)

    def entropy_symbol(self) -> float:
        """H(symbol) in bits: how much variety the symbols themselves carry.

        This is the ceiling on what any sentence built from these symbols can
        say about her: a dictionary cannot add information the symbol stream
        does not contain (data processing inequality).
        """
        n = self.total
        if n <= 0:
            return 0.0
        return -sum((k / float(n)) * math.log2(k / float(n))
                    for k in self.symbol_totals.values() if k > 0)

    def specificity(self, symbol: int) -> float:
        """Share of the context entropy the symbol removes: 1 - H(c|s)/H(c)."""
        h_c = self._entropy_context()
        if h_c <= 0.0:
            return 0.0
        row = self.counts.get(int(symbol))
        if not row:
            return 0.0
        n_s = float(sum(row.values()))
        h_cs = -sum((k / n_s) * math.log2(k / n_s) for k in row.values() if k > 0)
        return max(min(1.0 - h_cs / h_c, 1.0), 0.0)

    def referent(self, symbol: int) -> Optional[tuple[int, ...]]:
        """The context this symbol has most often occurred in."""
        row = self.counts.get(int(symbol))
        if not row:
            return None
        # Ties broken by the smaller bucket tuple, so a replay gives one answer.
        return min(row.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    def symbols_for(self, context: tuple[int, ...]) -> list[int]:
        """Reverse direction: which symbols occur in this context, most often first."""
        c = tuple(int(x) for x in context)
        hits = [(s, row[c]) for s, row in self.counts.items() if c in row]
        hits.sort(key=lambda sc: (-sc[1], sc[0]))
        return [s for s, _ in hits]

    # ---------- what L6 is allowed to see ----------

    def label(self, symbol: int) -> str:
        """Discrete referent label (no vector). "none" until it is measurable."""
        c = self.referent(int(symbol))
        if c is None:
            return "none"
        return "b" + "b".join(str(int(x)) for x in c)

    def coverage(self) -> float:
        """Share of symbols that have been observed at all."""
        if not self.symbol_totals:
            return 0.0
        return sum(1 for s in self.symbol_totals if self.counts.get(s)) / float(len(self.symbol_totals))

    # ---------- persistence ----------

    def to_dict(self) -> dict:
        """Plain data: JSON/ndarray friendly, keys as strings (carrier flattens)."""
        return {
            "buckets": int(self.buckets),
            "hist": int(self.hist),
            "counts": {str(s): {",".join(str(int(x)) for x in c): int(k)
                                for c, k in row.items()}
                       for s, row in self.counts.items()},
            "total": int(self.total),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroundingTable":
        t = cls(buckets=int(data.get("buckets", BUCKETS)),
                hist=int(data.get("hist", HIST)))
        for s, row in (data.get("counts") or {}).items():
            symbol = int(s)
            for key, k in row.items():
                ctx = tuple(int(x) for x in str(key).split(",") if str(x) != "")
                t._add(symbol, ctx, int(k))
        # `total` is derived by _add(); a stored value that disagrees is ignored
        # rather than trusted, so counts and totals cannot diverge.
        return t
