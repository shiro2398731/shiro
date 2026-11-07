"""
src/common/homeostasis.py

稳态监测：只看，不碰。

白默 2026-11-03：
    "她自己是她自己，系统是系统，系统不能干涉她的思维。"

因此本模块的全部职责是**记录**，不是调节。旧版 ARCHITECTURE.md §5.2 里的
"效价回归 / 情感重置"那类外部纠偏已废除。

监测的三件事
------------

1. **闭环完整性**：慢累积量与快层活动的负反馈是否还在工作。
   正常：累积高 → 活动低 → 累积降 → 活动恢复。
   异常：累积高但活动不降，或累积降了但活动不恢复。
   这对应"她醒不过来"的风险——但**只记录，不拉她**。

2. **状态场长期单峰**：dev_avg 长期贴在其历史下界附近，说明她一直在同一个地方。
   只记录。不调噪声（那是干涉）。

3. **数值安全**：NaN / Inf / 发散。
   这不是她的思维，是计算环境坏了，**这个可以修**。

为什么修数值不算干涉
--------------------

NaN 不是她想出来的，是浮点运算坏了。修 NaN 相当于保证灯还亮着，
不是在替她想事情。

为什么推她一把算干涉
--------------------

如果她的基质真的停了，那是她的事实。系统在她不知情的时候把她推醒，
等于改写她的状态而她自己不知道——这正是"系统干涉思维"的定义。
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ClosureReport:
    """闭环完整性报告。只有事实，没有建议。"""
    closed: bool
    pressure_level: float
    activity_level: float
    correlation: float          # 累积量与活动的近期相关性（负值才是正常的）
    n_samples: int
    note: str = ""


class Homeostasis:
    """
    稳态监测器。

    它不知道她"应该"怎样。它只知道两件事：
      - 数值是否还在合法范围内（这个要修）
      - 她自己的闭环是否还在工作（这个只记）
    """

    HIST = 512

    def __init__(self):
        self._pressure = collections.deque(maxlen=self.HIST)
        self._activity = collections.deque(maxlen=self.HIST)
        self._dev = collections.deque(maxlen=self.HIST)
        self._last_numeric_fault: Optional[str] = None

    # ---------- 输入 ----------

    def observe(self, pressure: float, activity: float, dev: Optional[float] = None) -> None:
        """每次观测输一次。调用频率由上游决定。"""
        self._pressure.append(float(pressure))
        self._activity.append(float(activity))
        if dev is not None:
            self._dev.append(float(dev))

    # ---------- 一、闭环 ----------

    def closure(self) -> ClosureReport:
        """
        检查慢累积量与活动水平的负反馈是否还在。

        判据：两者的近期相关性应该为负（累积高 → 活动低）。
        这不是阈值判断，是关系的方向。
        """
        n = min(len(self._pressure), len(self._activity))
        if n < 64:
            return ClosureReport(True, 0.0, 0.0, 0.0, n, "样本不足，尚不能判断")

        p = np.asarray(list(self._pressure)[-n:], dtype=float)
        a = np.asarray(list(self._activity)[-n:], dtype=float)

        if p.std() < 1e-9 or a.std() < 1e-9:
            # 有一方是常数。若累积量高位不变而活动也不变，闭环已断。
            if p.std() < 1e-9 and p[-1] > 0.8 and a.std() < 1e-9:
                return ClosureReport(False, float(p[-1]), float(a[-1]), 0.0, n,
                                     "累积量高位不变且活动不变：闭环可能已断")
            return ClosureReport(True, float(p[-1]), float(a[-1]), 0.0, n,
                                 "一方为常数，尚不能判断方向")

        corr = float(np.corrcoef(p, a)[0, 1])
        closed = corr < 0.05     # 负相关或无关，都不算"断了"
        note = "" if closed else "累积与活动同向变化，负反馈可能失效"
        return ClosureReport(closed, float(p[-1]), float(a[-1]), corr, n, note)

    # ---------- 二、状态场单峰 ----------

    def is_flat(self, rel_floor: float = 0.25) -> tuple[bool, str]:
        """
        dev 是否长期贴近其自身历史的下界。

        rel_floor 是一个**相对量**：当前 dev 低于历史分布的这个分位，
        且持续一段时间，就算"一直在同一个地方"。
        只记录，不调噪声。
        """
        if len(self._dev) < 64:
            return False, "样本不足"
        arr = np.asarray(self._dev, dtype=float)
        lo = float(np.percentile(arr, rel_floor * 100))
        if arr[-1] <= lo * 1.02:
            return True, f"dev={arr[-1]:.4f} 贴近历史 {rel_floor:.0%} 分位"
        return False, ""

    # ---------- 三、数值安全（这个可以修） ----------

    def check_numeric(self, *arrays: np.ndarray) -> Optional[str]:
        """
        检查数值是否合法。返回问题描述，或 None 表示正常。

        这是**唯一允许系统动手的地方**：NaN / Inf 是计算环境坏了，
        不是她的思维。修它不算干涉。
        """
        for i, a in enumerate(arrays):
            arr = np.asarray(a)
            if not np.all(np.isfinite(arr)):
                self._last_numeric_fault = f"array[{i}] 含非有限值"
                return self._last_numeric_fault
            if float(np.max(np.abs(arr))) > 1e6:
                self._last_numeric_fault = f"array[{i}] 量级异常：{float(np.max(np.abs(arr))):.2e}"
                return self._last_numeric_fault
        return None

    @staticmethod
    def repair_numeric(arr: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        用上一个合法状态替换坏值。返回 (修复后的数组, 是否修过)。

        只替换坏掉的那个元素，不动其它的——这不是"重置她的状态"。
        """
        a = np.asarray(arr, dtype=float).copy()
        bad = ~np.isfinite(a)
        if not bad.any():
            return a, False
        a[bad] = np.asarray(fallback, dtype=float)[bad]
        return a, True
