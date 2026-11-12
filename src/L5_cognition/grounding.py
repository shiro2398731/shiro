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

实测（她的真实存档，3000 步，12 个符号 × 16 个指代桶，n=2936；`python tools/grounding.py`）：
    MI(real) = 0.7395 bits，MI(shuffled) = 0.0463 bits —— 真实的指代信息是零假设的 16 倍。
即：她的区域编号不是随机的，它带着关于她自己状态的真实信息。

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
        n = self.total
        if n <= 0:
            return 0.0
        return -sum((k / float(n)) * math.log2(k / float(n))
                    for k in self.context_totals.values() if k > 0)

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
