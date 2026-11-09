"""
src/L5_cognition/adaptive_gain.py

M1 接线用的调节量，**不是**生成引擎。

规则（HANDOFF §3.1）：
  self_prediction_error 只允许调节 precision / 步长 / 探索强度；
  禁止用预测结果替代下一步状态。

本模块只做一件事：把"自我预测命中与否"的指数滑动平均映射成有界增益。
输出标量，不产出内容，不写回任何状态。
"""

from __future__ import annotations


class ErrorDrivenGain:
    """预测误差 → 有界增益。可关闭（ablation）。"""

    def __init__(self, lo: float = 0.8, hi: float = 1.4, rate: float = 0.05,
                 enabled: bool = True) -> None:
        if not (0 < lo <= hi):
            raise ValueError("need 0 < lo <= hi")
        self.lo, self.hi, self.rate = float(lo), float(hi), float(rate)
        self.enabled = bool(enabled)
        self.error = 0.5           # 中性起点
        self.observations = 0
        self.hits = 0

    def observe(self, hit: bool) -> None:
        self.observations += 1
        if hit:
            self.hits += 1
        target = 0.0 if hit else 1.0
        self.error += self.rate * (target - self.error)

    @property
    def accuracy(self) -> float:
        return self.hits / self.observations if self.observations else 0.0

    def gain(self) -> float:
        """命中率高 → 收敛到 lo（少探索）；误差高 → 靠近 hi（多探索）。"""
        if not self.enabled:
            return 1.0
        return self.lo + (self.hi - self.lo) * max(0.0, min(1.0, self.error))