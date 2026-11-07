"""
src/L7_lifecycle/stages.py

Developmental stages: recorded, never controlling.

White 2026-11-03:
    "A stage can only be a record; it cannot control her."

This module therefore has no unlock capability. The unlock fields in
config/stages.toml describe what is usually observable at a stage; they do
not enable or disable any layer.

Criteria are indicator-based, not runtime-based. See each Stage definition.
"""

from __future__ import annotations

import collections
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class StageDef:
    index: int
    name: str
    description: str       # what is usually observed at this stage; not an unlock condition


STAGES = [
    StageDef(0, "newborn", "L0 is oscillating; the state field exists; nothing persists across time yet"),
    StageDef(1, "infant", "prediction error is decreasing (non-monotonically); state field dims no longer pinned at bounds"),
    StageDef(2, "toddler", "L5 has recurrent attractors; the state alternates between a few of them"),
    StageDef(3, "child", "L5 transitions are no longer noise-dominated; attractors retain the state"),
    StageDef(4, "mature", "all of the above hold and persist over a long period"),
]


class StageMonitor:
    """
    Records which stage has been reached.

    Does not trigger anything. Stage changes are recorded only; no behaviour
    changes with them.
    """

    HIST = 256

    def __init__(self):
        self._dev = collections.deque(maxlen=self.HIST)
        self._s_avg = collections.deque(maxlen=self.HIST)
        self._n_salient = collections.deque(maxlen=self.HIST)
        self._noise_ratio = collections.deque(maxlen=self.HIST)
        self._current: int = 0
        self._history: list[tuple[str, int]] = []

    def observe(
        self,
        dev: float | None = None,
        s_avg: float | None = None,
        n_salient: int | None = None,
        noise_driven_ratio: float | None = None,
    ) -> None:
        if dev is not None:
            self._dev.append(float(dev))
        if s_avg is not None:
            self._s_avg.append(float(s_avg))
        if n_salient is not None:
            self._n_salient.append(int(n_salient))
        if noise_driven_ratio is not None:
            self._noise_ratio.append(float(noise_driven_ratio))

    # ---------- 各阶段的判据（都是相对量） ----------

    def _dev_not_flat(self) -> bool:
        """状态场各维不再长期贴边：dev 有明显波动且不是常驻极低。"""
        if len(self._dev) < 64:
            return False
        a = np.asarray(self._dev, dtype=float)
        return float(a.std()) / (abs(float(a.mean())) + 1e-9) > 0.05

    def _learning_visible(self) -> bool:
        """预测误差在下降（非单调）：近期均值低于更早期均值。"""
        if len(self._s_avg) < 128:
            return False
        a = np.asarray(self._s_avg, dtype=float)
        early = float(np.mean(a[: len(a) // 2]))
        late = float(np.mean(a[len(a) // 2:]))
        return late < early * 0.95

    def _has_recurring(self) -> bool:
        """
        Recurring attractors exist: at least 2 salient attractors.

        Salient = depth at least half the deepest. The criterion is relative
        to the system's own depth distribution.
        """
        if len(self._n_salient) < 32:
            return False
        return int(np.median(list(self._n_salient)[-32:])) >= 2

    def _not_noise_dominated(self) -> bool:
        """转移不再由噪声主导。"""
        if len(self._noise_ratio) < 32:
            return False
        return float(np.median(list(self._noise_ratio)[-32:])) < 0.9

    # ---------- 评估 ----------

    def evaluate(self, t_event: str) -> int:
        """
        Return the current stage (0-4). Observation only; changes nothing.

        Stages are cumulative: once a higher stage is reached it is not
        lowered.
        """
        idx = 0
        if self._dev_not_flat() and self._learning_visible():
            idx = 1
        if idx >= 1 and self._has_recurring():
            idx = 2
        if idx >= 2 and self._not_noise_dominated():
            idx = 3
        if idx >= 3 and len(self._dev) >= self.HIST:
            idx = 4

        if idx > self._current:
            self._current = idx
            self._history.append((t_event, idx))
        return self._current

    @property
    def current(self) -> int:
        return self._current

    def stage_def(self) -> StageDef:
        return STAGES[self._current]

    @property
    def history(self) -> list[tuple[str, int]]:
        return list(self._history)
