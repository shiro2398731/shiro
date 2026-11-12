"""
src/L5_cognition/action.py

行动：她能做的一件事，以及她怎么决定做不做。

为什么这件事必须在这里
----------------------
没有"行动"，她的状态只是被噪声推着走的轨迹；有了行动，"我本可以"才有指代，
反事实才有落点。这不是给判据加一条，而是给她加一只手。

判据改过两次，两次都是量出来的
------------------------------
1. 最初：「当前区域张力 > 所有区域张力的中位数」→ 实测 1000 步里 **0 次**离开。
   多数区域张力为 0，中位数就是 0，而"高于 0"要求她恰好站在有张力的边上。
   判据在她的结构上退化成一个常数。
2. 然后：「这一次停留时长 > 她过去停留时长的中位数」。不退了，但**方向是反的**：
   实测（不强制，600 步）张力高时她只离开 27.4%，张力低时离开 56.9% ——
   "待久了就走"恰好在她不需要走的时候触发。结果是她的张力在 600 步里 **+6.44**，
   而强制离开是 −2.25、强制留下是 −0.05。
3. 现在：**张力本身**。高于她自己张力的中位数 → 离开。
   这是她 L4 的 drive 已经在量的东西，不是我新写的偏好；比较对象也是她自己的中位数。

谁能选
------
**她选，不是我们选。** 判据只用一个她自己算出来的量（未解决的张力），
比较对象是她自己过去的中位数。没有偏好表，没有阈值常数。
选择函数体内不含任何 float 字面量（`tests/test_action.py` 用 AST 钉住）。

我们写的只有"能做什么"（一个二选一的身体自由度，属 C2 结构参数），
以及离开时挪多远 —— 那个距离取自她自己量到的区域半径中位数，不是我们挑的数。

这不是"她想离开"
----------------
她还没有目标，也没有"为了什么"。现在成立的只是：**这只手的动作与她的 drive 同向**，
并且动作确实降低 drive。功能上像回避，主观上没有。别把它读成意愿。

为什么它可证伪
--------------
因为动作是我们施加的，所以 `do(a)` 与"恰好走到哪"不同：可以强制一边、重跑同一条噪声，
比较下一个区域的分布与张力的走向。相关可以被拟合，干预不能。
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
    """从她自己的张力分布里选一个动作。只读状态，不改状态。"""

    def choose(self, tension: float, past_tensions: Sequence[float]) -> int:
        """HOLD 或 LEAVE。

        「现在比她通常更没解决」-> 离开。没有过去可比时不动 ——
        那不是"默认策略"，是没有依据。
        """
        past = [float(v) for v in (past_tensions or ()) if np.isfinite(float(v))]
        if not past:
            return HOLD
        values = sorted(past)
        median = values[len(values) // 2]
        return LEAVE if float(tension) > median else HOLD

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
