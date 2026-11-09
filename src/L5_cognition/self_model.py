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