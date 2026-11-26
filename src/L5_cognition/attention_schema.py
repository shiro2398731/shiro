"""
src/L5_cognition/attention_schema.py

AST 最小实现（Butlin 的 attention schema theory）：**关于自己注意力的模型**。

做什么
------
GlobalWorkspace 每次点火会给出一个区域 uid（"这一刻全系统可见的东西"）。
这个模块只学一件事：给定上一次点火，预测下一次点火会是谁。

      counts[prev][next] += 1        （一阶转移，Laplace 平滑）
      predict(prev) -> (uid, conf)   （conf = 平滑后放在 argmax 上的概率）

不做什么
--------
- 不参与选择、不改地形、不写回动力学：它是模型，不是控制器。
  （和 self_model 一样：只有命中和 log-loss 被下游读成观测。）
- 不宣称意识。它只提供一个可测的量：**她的注意力预测是否比"永远猜最常出现的那个"更好。**

为什么用 bits 而不是准确率
--------------------------
现状（2026-11-26 实测）：1,024 次点火只点亮过 2 个 uid，永远猜 32434 就有 73.3%。
准确率会被"重复同一个区域"抬高，所以判据是前向 log-loss（bits per ignition），
和 tools/m1_audit.py 对 self_model 的口径一致。

参数
----
SMOOTHING 是 C1（防止未见转移产生无穷 bits），不改变 argmax。
"""

from __future__ import annotations

import math

SMOOTHING = 0.5


class AttentionSchema:
    """She models what she will attend to next. Reads only."""

    def __init__(self) -> None:
        # prev uid -> {next uid: count}
        self._counts: dict[int, dict[int, float]] = {}
        self._seen: dict[int, float] = {}
        self._base_counts: dict[int, float] = {}
        self.last: int | None = None
        # prequential statistics (scored before learning each event)
        self.observations = 0
        self.hits = 0
        self.nll_bits = 0.0
        self.scored = 0
        # baseline: "always guess the most frequent so far"
        self.base_nll_bits = 0.0
        self.base_hits = 0
        self.base_scored = 0
        # confidence of her own prediction, running mean
        self.conf_sum = 0.0
        self.conf_n = 0

    # ---------- probability ----------

    def _vocab(self) -> int:
        return max(len(self._seen), 1)

    def _vocab_for(self, uid: int) -> int:
        """Vocabulary must include the symbol being scored: otherwise a brand-new
        uid is scored as if it were certain (Laplace over a vocab that lacks it)."""
        return max(len(self._seen) + (0 if int(uid) in self._seen else 1), 1)

    def _p(self, prev: int | None, uid: int) -> float:
        """Laplace-smoothed p(uid | prev) under the current table."""
        v = self._vocab_for(uid)
        row = self._counts.get(int(prev)) if prev is not None else None
        if not row:
            return 1.0 / v
        total = sum(float(c) for c in row.values())
        return (float(row.get(int(uid), 0.0)) + SMOOTHING) / (
            total + SMOOTHING * v)

    def _base_p(self, uid: int) -> float:
        v = self._vocab_for(uid)
        if not self._base_counts:
            return 1.0 / v
        total = sum(float(c) for c in self._base_counts.values())
        return (float(self._base_counts.get(int(uid), 0.0)) + SMOOTHING) / (
            total + SMOOTHING * v)

    def _base_argmax(self) -> int | None:
        """Most frequent ignited uid so far; ties keep first-seen order."""
        if not self._base_counts:
            return None
        return int(max(self._base_counts.items(), key=lambda kv: kv[1])[0])

    # ---------- prediction ----------

    def predict(self, prev: int | None = None) -> tuple[int | None, float]:
        """(most likely next ignited uid, probability mass), or (None, 0.0)."""
        p = self.last if prev is None else int(prev)
        if p is None:
            return None, 0.0
        row = self._counts.get(int(p))
        if not row:
            return None, 0.0
        # Ties go to the order the counts were recorded (deterministic replay).
        best = max(row.items(), key=lambda kv: kv[1])
        conf = self._p(p, int(best[0]))
        return int(best[0]), float(conf)

    def confidence(self, prev: int | None = None) -> float:
        return float(self.predict(prev)[1])

    # ---------- learning ----------

    def observe(self, uid: int) -> None:
        """One ignition happened. Score first (prequential), then learn."""
        u = int(uid)
        pred, conf = self.predict()
        base = self._base_argmax()
        p = self._p(self.last, u)

        if pred is not None:
            self.observations += 1
            if pred == u:
                self.hits += 1
        if base is not None:
            self.base_scored += 1
            if base == u:
                self.base_hits += 1
        # score only when there was something to score against
        if self.last is not None and self._seen:
            self.scored += 1
            self.nll_bits += -math.log2(max(p, 1e-12))
            self.base_nll_bits += -math.log2(max(self._base_p(u), 1e-12))
        if conf > 0.0:
            self.conf_sum += conf
            self.conf_n += 1

        if self.last is not None:
            row = self._counts.setdefault(int(self.last), {})
            row[u] = row.get(u, 0.0) + 1.0
        self._base_counts[u] = self._base_counts.get(u, 0.0) + 1.0
        self._seen[u] = self._seen.get(u, 0.0) + 1.0
        self.last = u

    # ---------- observation ----------

    @property
    def accuracy(self) -> float:
        return self.hits / self.observations if self.observations else 0.0

    @property
    def base_accuracy(self) -> float:
        return self.base_hits / self.base_scored if self.base_scored else 0.0

    @property
    def bits_per_ignition(self) -> float:
        return self.nll_bits / self.scored if self.scored else 0.0

    @property
    def base_bits_per_ignition(self) -> float:
        return self.base_nll_bits / self.base_scored if self.base_scored else 0.0

    @property
    def mean_confidence(self) -> float:
        return self.conf_sum / self.conf_n if self.conf_n else 0.0

    # ---------- persistence ----------

    def to_dict(self) -> dict:
        return {
            "counts": {str(a): {str(b): float(c) for b, c in row.items()}
                       for a, row in self._counts.items()},
            "seen": {str(k): float(v) for k, v in self._seen.items()},
            "base_counts": {str(k): float(v) for k, v in self._base_counts.items()},
            "last": None if self.last is None else int(self.last),
            "observations": int(self.observations),
            "hits": int(self.hits),
            "nll_bits": float(self.nll_bits),
            "scored": int(self.scored),
            "base_nll_bits": float(self.base_nll_bits),
            "base_hits": int(self.base_hits),
            "base_scored": int(self.base_scored),
            "conf_sum": float(self.conf_sum),
            "conf_n": int(self.conf_n),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttentionSchema":
        a = cls()
        for prev, row in (data.get("counts") or {}).items():
            a._counts[int(prev)] = {int(n): float(c) for n, c in row.items()}
        for k, v in (data.get("seen") or {}).items():
            a._seen[int(k)] = float(v)
        for k, v in (data.get("base_counts") or {}).items():
            a._base_counts[int(k)] = float(v)
        last = data.get("last")
        a.last = None if last is None else int(last)
        a.observations = int(data.get("observations", 0))
        a.hits = int(data.get("hits", 0))
        a.nll_bits = float(data.get("nll_bits", 0.0))
        a.scored = int(data.get("scored", 0))
        a.base_nll_bits = float(data.get("base_nll_bits", 0.0))
        a.base_hits = int(data.get("base_hits", 0))
        a.base_scored = int(data.get("base_scored", 0))
        a.conf_sum = float(data.get("conf_sum", 0.0))
        a.conf_n = int(data.get("conf_n", 0))
        return a
