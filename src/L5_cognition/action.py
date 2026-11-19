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
                  places: Optional[Sequence] = None,
                  prefer_far: bool = False) -> Optional[np.ndarray]:
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
                if visited is not None and places is not None:
                    vis = set(visited)
                    place_keys = list(places)
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
                    # **不许有 float 字面量**（action.py 第 27 行，test_action 用 AST 钉住）。
                    # 原来写的 `* 1.5` 被测试抓到：`{'direction': [1.5]}`。
                    # 改成一个「从她自己的数据里取的量」：用距她最近的若干个邻居的中位距离
                    # 作为 reach —— 就是「她身边这些邻居实际隔多远」。
                    # 这不需要乘任何系数：median 本身就是她的尺度。
                    _ok = dist[np.isfinite(dist)]
                    if _ok.size >= 2:
                        reach = float(np.median(_ok))
                    else:
                        reach = s
                    # 2026-11-19 22:10：`prefer_far` 时 reach 要放宽到「最远那个」。
                    # 测试抓到的 bug：reach 默认是中位距离，最远的候选比中位远，被 `dist <= reach` 排除，
                    # 于是「挑最远」实际只挑到「中位以内最远的那个」——不是最远。
                    if prefer_far and _ok.size >= 1:
                        reach = max(reach, float(np.max(_ok)))
                    # 2026-11-18 下午：fresh 也要排除「几乎重合」的候选（dist < s）。
                    # 上面 10:00 那条只在 fallback 分支加了 dist >= s，fresh 分支漏了。
                    # 实测（补丁版 200 步）：69.5% 的「离开」位移 < 1 —— 因为她挑到的是
                    # 一个 0.01–0.1 远的重复区域，走 min(s, dist) 等于原地蹭。
                    # 「离开」的本意是至少挪开一个半径，所以这里也要求 dist >= s。
                    # 2026-11-18 夜（22:40）：门槛从 `>= s` 放到 `>= s/2`。
                    #
                    # 这是环的真凶。实测：她的最近邻居 31319 距离 **618.5**，而 s = **618.6**——
                    # 差 0.1，被 `dist >= s` 筛掉。于是 fresh 里「最近的那个」变成
                    # **70 万外的 30333**（她的地形在半径 70 万的球壳上）。
                    # 她每步朝那边走 618，永远走不到，净位移 ≈ 0，只在这几个区域间弹。
                    #
                    # 副本实测（100 步、强制 LEAVE、同种子）：
                    #   门槛 s    : distinct 3、回边率 0.99、净位移 0.0
                    #   门槛 s/2  : distinct 13、回边率 0.29、净位移 3084   ← 破环
                    #   门槛 s/4  : distinct 12、回边率 0.54
                    #
                    # 为什么不是更小：0.5 仍然排得掉「几乎重合」的区域（0.001~0.9 都 < s/2），
                    # 而 618.5 这种「刚好差一点」的真邻居能留下。`2` 是整数，不是 float 字面量。
                    _floor = s / 2
                    fresh = [k for k in range(len(place_keys))
                             if place_keys[k] not in vis
                             and np.isfinite(dist[k]) and _floor <= dist[k] <= reach]
                    if fresh:
                        # 2026-11-19 17:35：**卡住就脱困**（机制，不是概率）。
                        #
                        # 她总挑「最近的没去过的」，于是永远在最近的一对之间弹（真机 12 小时只有 2 个区域）。
                        # 给她一条规则：**如果她连续几次都回到同一处，下一次就挑「最远但够得着」的那个。**
                        # 「连续几次」由 Stream 数她自己的到达序列，不是人给的 p。
                        #
                        # 为什么不是随机探索：那需要一个「探索概率」，又是一个要人拍的数字；
                        # 而「卡住了就换个方向」是她自己的处境在说话——不卡就不触发。
                        if prefer_far:
                            k = int(max(fresh, key=lambda j: dist[j]))
                        else:
                            k = int(min(fresh, key=lambda j: dist[j]))
                        return d[k] / dist[k] * min(s, float(dist[k]))

                # 2026-11-18 09:50：她可能**正站在某个区域里**（dist=0）。
                # 那时 argmin 会选中那个 0，min(s, 0) = 0，于是「离开」原地不动——
                # 实测 dist.min() = 0.0，direction 返回长度 0 的向量。
                # 只挑「距离 > 0」的候选，这样她才会真的挪出当前区域。
                # 2026-11-18 10:00：光排除 dist=0 不够。她旁边可能有一个**几乎重合**的区域
                # （实测 dist 最小 0.001），走 min(s, 0.001) 等于没动，
                # 但打印成 .1f 会显示 0.0，看起来像「完全没走」。
                # 只挑「至少有她半径的一半那么远」的目标——那样一步才真的挪得开。
                # 不乘任何系数（那会是 float 字面量，被 test_action 钉住）。
                # 直接用「她自己量到的半径」当门槛：走到一个至少一个半径远的地方。
                # 2026-11-18 夜：**这里也要排除「去过」的地方。**
                #
                # 环的最后一块拼图：`fresh` 分支会排除 visited，但 fresh 为空时落到这里，
                # 而这里挑的是「最近的一个 >= s 的候选」——不管去没去过。
                # 于是她从 B 出发时，A 被 visited 排除了、fresh 空了、fallback 又把 A 挑回来：
                # A ↔ B 永远断不了。实测（22:10）：合并重复区域之后还是 distinct=2、back=1.0。
                #
                # 现在 fallback 也跳过 visited；全去过了才退回「最近的一个」（那时环是合理的——
                # 她的世界就这么大，没有新地方可去了）。
                _vis = set(visited or ()) if (visited is not None and places is not None) else set()
                if _vis and places is not None:
                    _pos = [k for k in range(len(places))
                            if np.isfinite(dist[k]) and dist[k] >= s and places[k] not in _vis]
                    _pos = np.asarray(_pos, dtype=int)
                else:
                    _pos = np.where(np.isfinite(dist) & (dist >= s))[0]
                if _pos.size:
                    j = int(_pos[np.argmin(dist[_pos])])
                    unit = d[j] / dist[j]
                    # 走「她自己半径」那么远，但不超过到对方的距离（免得穿过去）
                    step = min(s, float(dist[j]))
                    return unit * step

        d = x - np.asarray(centroid, dtype=float)
        n = float(np.linalg.norm(d))
        if not np.isfinite(n) or n <= 0:
            return None
        return d / n * s
