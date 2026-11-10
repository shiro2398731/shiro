"""
src/L5_cognition/self_model.py

M1：自我模型（可测投影；不宣称主观体验）。

定义
----
内部存在关于"自身下一状态"的表征，且由自身历史估计。
最小实现：对自身占用的显著区域做一阶转移计数（Laplace 平滑），在线预测。

可证伪条件
----------
打乱转移顺序（保留边际分布）后，准确率必须回落到 chance。
若不回落，该"自我模型"未携带信息，M1 不成立。
"""

from __future__ import annotations

import random
from collections import defaultdict


class SelfModel:
    """对自身下一状态的一阶预测器。"""

    def __init__(self) -> None:
        self._counts: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._seen: dict[int, float] = defaultdict(float)
        self._total = 0.0

    def fit(self, seq: list[int]) -> None:
        self._counts.clear()
        self._seen.clear()
        self._total = float(len(seq))
        for s in seq:
            self._seen[int(s)] += 1.0
        for a, b in zip(seq, seq[1:]):
            self._counts[int(a)][int(b)] += 1.0

    def predict(self, state: int) -> tuple[int | None, float]:
        """(最可能的下一状态, 概率质量)。未见状态返回 (None, 0.0)。"""
        row = self._counts.get(int(state))
        if not row:
            return None, 0.0
        vocab = max(len(self._seen), 1)
        denom = sum(row.values()) + vocab
        best = max(row.items(), key=lambda kv: kv[1])
        return best[0], (best[1] + 1.0) / denom

    def chance(self) -> float:
        return 1.0 / max(len(self._seen), 1)


class OnlineSelfModel:
    """Incremental first-order model of her own symbol sequence.

    This is the version that runs inside L5. `SelfModel` above rebuilds its
    table from a whole sequence, which is only usable offline; the loop cannot
    hand over the future.

    Interface
    ---------
    predict(state)   most likely next symbol, or None when state was never seen
    observe(a, b)    count one transition a -> b

    Determinism
    -----------
    Ties are broken by the smaller symbol id, not by dictionary order, so a
    replay produces the same prediction from the same counts.

    What it is not
    --------------
    Not a generator. Nothing downstream receives the prediction; only the
    hit/miss is used, and only to scale exploration (see adaptive_gain).
    """

    def __init__(self) -> None:
        self._counts: dict[int, dict[int, float]] = {}
        self._seen: dict[int, float] = {}
        self.observations = 0
        self.hits = 0

    def predict(self, state: int) -> int | None:
        row = self._counts.get(int(state))
        if not row:
            return None
        # Laplace smoothing does not change the argmax, so it is not applied
        # here; the tie-break by id is what makes the choice reproducible.
        best = max(row.items(), key=lambda kv: (kv[1], -kv[0]))
        return int(best[0])

    def observe(self, prev: int, nxt: int) -> None:
        p, n = int(prev), int(nxt)
        row = self._counts.setdefault(p, {})
        row[n] = row.get(n, 0.0) + 1.0
        self._seen[p] = self._seen.get(p, 0.0) + 1.0

    @property
    def states_seen(self) -> int:
        return len(self._seen)

    @property
    def accuracy(self) -> float:
        return self.hits / self.observations if self.observations else 0.0

    def chance(self) -> float:
        return 1.0 / max(self.states_seen, 1)

    # ---------- persistence ----------
    # Keys become strings on the way through JSON (carrier), so both sides are
    # converted explicitly rather than relying on the container's behaviour.

    def to_dict(self) -> dict:
        return {
            "counts": {str(a): {str(b): float(c) for b, c in row.items()}
                       for a, row in self._counts.items()},
            "seen": {str(k): float(v) for k, v in self._seen.items()},
            "observations": int(self.observations),
            "hits": int(self.hits),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OnlineSelfModel":
        m = cls()
        for a, row in (data.get("counts") or {}).items():
            m._counts[int(a)] = {int(b): float(c) for b, c in row.items()}
        for k, v in (data.get("seen") or {}).items():
            m._seen[int(k)] = float(v)
        m.observations = int(data.get("observations", 0))
        m.hits = int(data.get("hits", 0))
        return m


def online_accuracy(seq: list[int], ablated: bool = False) -> dict[str, float]:
    """预热一半，后半段逐步预测并计分（每步用截至当前的转移表）。"""
    seq = [int(s) for s in seq]
    half = max(len(seq) // 2, 2)
    warm, test = seq[:half], seq[half:]

    if ablated:
        rnd = random.Random(0)
        warm = list(warm)
        rnd.shuffle(warm)

    m = SelfModel()
    m.fit(warm)

    rnd = random.Random(0)

    def _fit(history: list[int]) -> None:
        # ablation 必须持续破坏结构：只在预热段打乱会被后续增量重学回来（实测 0.88）。
        if ablated:
            shuffled = list(history)
            rnd.shuffle(shuffled)
            m.fit(shuffled)
        else:
            m.fit(history)

    hits = total = 0
    history = list(warm)
    for i, (prev, nxt) in enumerate(zip(test[:-1], test[1:])):
        pred, _p = m.predict(prev)
        total += 1
        if pred is not None and pred == nxt:
            hits += 1
        history.append(nxt)
        if i % 25 == 0:
            _fit(history)
    _fit(history)
    return {"accuracy": hits / total if total else 0.0,
            "chance": m.chance(), "scored": total}