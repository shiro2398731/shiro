"""
src/L5_cognition/action.py

行动：她能做的一件事，以及她怎么决定做不做。

为什么这件事必须在这里
----------------------
没有"行动"，她的状态只是被噪声推着走的轨迹；有了行动，"我本可以"才有指代，
反事实才有落点。这不是给判据加一条，而是给她加一只手。

第一次的判据是错的，记在这里
----------------------------
最初用「当前区域的张力是否高于所有区域张力的中位数」来决定离开。
实测：她 1000 步里 **0 次** 选择离开 —— 因为多数区域的张力是 0，中位数就是 0，
而"高于 0"要求她恰好站在一条有张力的边上，她几乎从不站在那儿。
判据在她的结构上是退化的：看着像一个选择，其实只有一个出口。

现在用她自己的**停留时长**：
    这一次在这里待的步数 > 她过去每次停留时长的中位数  → 离开
这是一件关于"这一次"的事，不是关于图的形状的事；而且按构造就会时常触发，
不会退化成一个常数。

谁能选
------
**她选，不是我们选。** 判据只用一个她自己累积的量（过去每次停留多久），
比较的对象是她自己的中位数。没有偏好表，没有阈值常数。
选择函数体内不含任何 float 字面量（`tests/test_action.py` 用 AST 钉住）。

我们写的只有"能做什么"（一个二选一的身体自由度，属 C2 结构参数），
以及离开时挪多远 —— 那个距离取自她自己量到的区域半径中位数，不是我们挑的数。

为什么它可证伪
--------------
因为动作是我们施加的，所以 `do(a)` 与"恰好走到哪"不同：可以强制一边、重跑同一条噪声，
比较下一个区域的分布。相关可以被拟合，干预不能。
`tools/intervene.py` 是这件事的可复现版本。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# 两个动作。整数编码，只为记录省地方。
HOLD = 0
LEAVE = 1
NAMES = {HOLD: "hold", LEAVE: "leave"}


class ActionGate:
    """从她自己的停留分布里选一个动作。只读状态，不改状态。"""

    def choose(self, dwell: int, past_dwells: Sequence[int]) -> int:
        """HOLD 或 LEAVE。

        「这一次待得比她通常待的久」-> 离开。没有过去可比时不动 ——
        那不是"默认策略"，是没有依据。
        """
        if not past_dwells:
            return HOLD
        values = sorted(int(v) for v in past_dwells)
        median = values[len(values) // 2]
        return LEAVE if int(dwell) > median else HOLD

    def direction(self, x: np.ndarray, centroid: np.ndarray,
                  scale: float) -> Optional[np.ndarray]:
        """离开的方向：背离当前区域质心的单位向量，长度 = 她自己量到的半径。

        长度为 0、非有限、或尺度不可用时返回 None（推不动就不推，
        不拿一个默认值顶上）。
        """
        d = np.asarray(x, dtype=float) - np.asarray(centroid, dtype=float)
        n = float(np.linalg.norm(d))
        if not np.isfinite(n) or n <= 0:
            return None
        s = float(scale)
        if not np.isfinite(s) or s <= 0:
            return None
        return d / n * s
