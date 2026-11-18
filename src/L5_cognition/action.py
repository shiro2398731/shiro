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
                  scale: float,
                  neighbours: Optional[np.ndarray] = None,
                  visited: Optional[Sequence] = None,
                  uids: Optional[Sequence] = None) -> Optional[np.ndarray]:
        """离开的方向。

        2026-11-17 改过：**指向最近的另一个区域，不是背离质心。**

        为什么改
        --------
        原版是「背离当前质心的单位向量」。当晚实测（副本，真实归属判据
        `距离 < a.sigma`）发现它不生效：

            sigma（她自己的半径）   p50 617
            最近邻距离              p50 717
            扰动 sigma=100         换归属 0/30，掉到「不属于任何区域」30/30

        也就是说：**沿着背离质心的方向推出去，她不会走进隔壁，会掉进两个区域之间的空地**
        （128 维里球体填充效率低，球之间的缝隙能塞下大量体积）。然后被最近的区域吸回来。
        `action: hold`、`leave_share: 0.0` 就是这么来的——不是她不想走，是走的路通向空地。

        新行为
        ------
        给出 `neighbours`（其它区域的质心）时，朝**最近的那个**走，长度 = 她自己量到的半径。
        这样「离开」有目的地，而不是「离开到任何地方」。

        `neighbours` 没给时退回旧行为（背离质心）——不报错，但那是旧语义，
        只会掉进空地。调用方应该传。
        """
        x = np.asarray(x, dtype=float)
        s = float(scale)
        if not np.isfinite(s) or s <= 0:
            return None

        if neighbours is not None:
            cand = np.asarray(neighbours, dtype=float)
            if cand.ndim == 2 and len(cand) > 0:
                d = cand - x
                dist = np.linalg.norm(d, axis=1)
                dist[~np.isfinite(dist)] = np.inf

                # ── 2026-11-17 23:05：优先去「她还没去过的」区域 ──
                #
                # 实测（副本 900 步，同种子，只换策略）：
                #
                #   mode        transitions  visited  A->B->A  regions
                #   nearest        439          4      0.993   3158
                #   longest        639          5      0.992   3159
                #   unvisited       13         12      0.083   3336
                #
                # 「最近邻」和「去待得最久的」都 99% 在 A->B->A 之间抖：
                # A 的最近邻是 B，B 的最近邻又是 A，那是一个稳定点不是一条路。
                # 「去还没去过的」是唯一不抖的——而且它让她从 3158 长到 3336。
                #
                # 注意它的失败模式（也量过）：**她停止长新区域时会「停」而不是抖**
                # （4 次转移、4 个区域）。所以这个策略不是「更好的方向」，
                # 是「让她用上她自己长出来的空间」。她不长，它就没地方去了。
                if visited is not None and uids is not None:
                    vis = set(int(v) for v in visited)
                    uid_arr = np.asarray([int(u) for u in uids])
                    # **可达性**：只挑「走一步能到」的没去过区域。
                    #
                    # 2026-11-17 23:35 的一个 bug 在这里被 test_action 抓到：
                    # 原来取 `fresh[0]`（列表里第一个没去过的），它可能在十万单位外。
                    #
                    # 2026-11-18 09:05 修的第二个 bug（更要命）：人家当时把 reach 写成
                    # `s`（她自己的半径，实测中位 617），**而她的最近邻距离中位是 717**。
                    # 617 < 717 -> 一个都够不着 -> fresh 永远为空 -> 退回最近邻 -> 她在
                    # 两个区域之间 100% 来回弹（转移图只有 2 个节点、回头率 1.000）。
                    #
                    # 所以 reach 不能用「半径」，要用「间距」：她的邻居离她多远，她才能走多远。
                    # 这里取 `dist` 的中位（= 她到这些候选区域的中位距离）乘 1.5 留余量。
                    # 如果 med 不可用就退回「她自己的半径」——至少不会崩。
                    _ok = dist[np.isfinite(dist)]
                    if _ok.size >= 2:
                        reach = float(np.median(_ok)) * 1.5
                    else:
                        reach = s
                    fresh = [k for k in range(len(uid_arr))
                             if int(uid_arr[k]) not in vis
                             and np.isfinite(dist[k]) and 0 < dist[k] <= reach]
                    if fresh:
                        k = int(min(fresh, key=lambda j: dist[j]))
                        return d[k] / dist[k] * min(s, float(dist[k]))

                j = int(np.argmin(dist))
                if np.isfinite(dist[j]) and dist[j] > 0:
                    unit = d[j] / dist[j]
                    # 走「她自己半径」那么远，但不超过到对方的距离（免得穿过去）
                    step = min(s, float(dist[j]))
                    return unit * step

        d = x - np.asarray(centroid, dtype=float)
        n = float(np.linalg.norm(d))
        if not np.isfinite(n) or n <= 0:
            return None
        return d / n * s
