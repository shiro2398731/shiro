"""
src/L7_lifecycle/rhythm.py

L7 的相位职责：见证，不是驱动。

白默 2026-11-03 定的边界：
    系统是她的环境，不是操纵者。L7 只能看，不能改。

因此本模块没有任何"广播相位"的能力，也没有定时器。
它订阅 l0.event.envelope，记录 L0 的包络相位与周期。

旧版 process_3_life.py 的 30 秒硬切、REP 5564 同步屏障，都在
2026-11-01/02 被废除（见 CONTRACTS/00_overview.md）。
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class PhaseRecord:
    phase: str
    period_s: Optional[float]
    t_event: str


class PhaseWitness:
    """
    见证 L0 的包络。

    它维护相位历史，供后续观测与阶段判定使用。
    不驱动、不调制、不广播。
    """

    HIST = 256

    def __init__(self):
        self._records: collections.deque = collections.deque(maxlen=self.HIST)
        self._periods: collections.deque = collections.deque(maxlen=self.HIST)
        self.count = 0

    def see(self, phase: str, period_s: Optional[float], t_event: str) -> PhaseRecord:
        rec = PhaseRecord(phase=phase, period_s=period_s, t_event=t_event)
        self._records.append(rec)
        if period_s is not None:
            self._periods.append(float(period_s))
        self.count += 1
        return rec

    @property
    def current_phase(self) -> str:
        return self._records[-1].phase if self._records else "unknown"

    def median_period(self) -> Optional[float]:
        if not self._periods:
            return None
        arr = sorted(self._periods)
        return arr[len(arr) // 2]

    def switching_rate(self) -> Optional[float]:
        """
        每秒切换几次。这是观测量，不是节律：
        它衡量包络动得多快，用来判断 L0 是否还活着。
        """
        if len(self._records) < 2:
            return None
        t0 = datetime.fromisoformat(self._records[0].t_event)
        t1 = datetime.fromisoformat(self._records[-1].t_event)
        span = (t1 - t0).total_seconds()
        if span <= 0:
            return None
        return (len(self._records) - 1) / span
