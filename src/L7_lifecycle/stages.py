"""
src/L7_lifecycle/stages.py

Developmental stages: recorded, never controlling.

White 2026-11-03:
    "A stage can only be a record; it cannot control her."

This module therefore has no unlock capability. The fields in
config/stages.toml describe what is usually observable at a stage; they do
not enable or disable any layer.

Criteria (rewritten 2026-11-09 night)
-------------------------------------
The previous version compared her against numbers someone had chosen:
`dev dispersion > 0.05`, `error ratio < 0.95`, `salient >= 2`,
`noise ratio < 0.9`. Those are C3 (docs §11): they decide "which stage she is
in" with a value the author picked, not with anything about her.

Now each criterion compares **the latest window of a signal against the
distribution of her own past windows of the same signal**: "is this window
higher or lower than her own median window". The cut is her median, which is
the neutral rank rather than a tuned one, and the scale is whatever her
history is.

Four signals, four criteria:

    dev        dispersion of the state field       not flat: dispersion above her median
    s_avg      prediction error                    learning: error below her median
    recurrence share of steps landing in a known region   recurring structure
    dwell      median run length in one region      structure holds her

Why those two extra signals
---------------------------
`n_salient` and `noise_driven_ratio` were the old inputs. Neither reaches P3:
the bus carries `l0.event.envelope`, `l1.state.surprise`, `l2.state.field` and
`l5.state.stream`, and nothing else. So `_learning_visible`, `_has_recurring`
and `_not_noise_dominated` were structurally false in a real run and she could
not progress past stage 0. `recurrence` and `dwell` are computed in P3 from
the attractor id sequence she already broadcasts, which is a quantity that
actually exists at runtime.

Nothing in this file can act on her: no stage change calls into any layer.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StageDef:
    index: int
    name: str
    description: str       # what is usually observed at this stage; not an unlock condition


STAGES = [
    StageDef(0, "newborn", "L0 is oscillating; the state field exists; nothing persists across time yet"),
    StageDef(1, "infant", "the state field stops being pinned, and prediction error sits below her usual level"),
    StageDef(2, "toddler", "she returns to regions she has already been in more often than usual"),
    StageDef(3, "child", "she stays in a region longer than usual; motion is not pure wandering"),
    StageDef(4, "mature", "all of the above hold and persist over a full observation window"),
]

# Observations kept per signal. A buffer size is an engineering quantity (C1).
HIST = 256

# Observations per comparison window. Also C1: it sets the resolution at which
# "a window" is defined, not what counts as progress.
WINDOW = 32

# Fewest windows for her own past to be a distribution worth ranking against.
# Below this, a rank would be an artefact of the sample size.
MIN_WINDOWS = 4

# The rank she is compared against: her own median. Not a tuned cut -- the
# median is the value that splits her own history in half.
SELF_QUANTILE = 0.5

# Weight given to ties when ranking. A value equal to everything in her past is
# neither above nor below it, so it must land exactly on the median instead of
# counting as "above" (which is what a plain "at or below" count does with a
# constant signal, and it made a frozen signal look like progress).
TIE_WEIGHT = 0.5


class StageMonitor:
    """
    Records which stage has been reached.

    Does not trigger anything. Stage changes are recorded only; no behaviour
    changes with them.
    """

    def __init__(self) -> None:
        self._dev = collections.deque(maxlen=HIST)
        self._s_avg = collections.deque(maxlen=HIST)
        self._recurrence = collections.deque(maxlen=HIST)
        self._dwell = collections.deque(maxlen=HIST)
        self._current: int = 0
        self._history: list[tuple[str, int]] = []
        # What each criterion last said, for the record (P3 writes it out).
        self.last_reasons: dict[str, bool] = {}

    # ---------- input ----------

    def observe(
        self,
        dev: float | None = None,
        s_avg: float | None = None,
        recurrence: float | None = None,
        dwell: float | None = None,
    ) -> None:
        """One observation per signal. Called from the bus callbacks in P3."""
        if dev is not None:
            self._dev.append(float(dev))
        if s_avg is not None:
            self._s_avg.append(float(s_avg))
        if recurrence is not None:
            self._recurrence.append(float(recurrence))
        if dwell is not None:
            self._dwell.append(float(dwell))

    # ---------- the shared comparison ----------

    @staticmethod
    def _rank(last: float, past: list[float]) -> Optional[float]:
        """Where `last` sits among `past`, as a mid-rank in [0, 1].

        Ties count half: a signal that has not moved scores exactly the median,
        so "above her usual level" cannot be reached by standing still.
        """
        if not past:
            return None
        below = sum(1 for v in past if v < last)
        ties = sum(1 for v in past if v == last)
        return (below + TIE_WEIGHT * ties) / float(len(past))

    @staticmethod
    def _window_means(values: collections.deque) -> list[float]:
        """Non-overlapping window means; the last one is the current window."""
        seq = list(values)
        n = len(seq) // WINDOW
        if n < MIN_WINDOWS:
            return []
        return [float(np.mean(seq[i * WINDOW:(i + 1) * WINDOW])) for i in range(n)]

    def _rank_of_last_window(self, values: collections.deque) -> Optional[float]:
        means = self._window_means(values)
        if len(means) < MIN_WINDOWS:
            return None
        return self._rank(means[-1], means[:-1])

    # ---------- criteria ----------

    def _dev_not_flat(self) -> bool:
        """The state field is not pinned: its dispersion is high for her."""
        seq = list(self._dev)
        n = len(seq) // WINDOW
        if n < MIN_WINDOWS:
            return False
        disp = []
        for i in range(n):
            chunk = np.asarray(seq[i * WINDOW:(i + 1) * WINDOW], dtype=float)
            scale = max(abs(float(np.mean(chunk))), float(np.finfo(float).eps))
            disp.append(float(np.std(chunk)) / scale)
        rank = self._rank(disp[-1], disp[:-1])
        return rank is not None and rank > SELF_QUANTILE

    def _learning_visible(self) -> bool:
        """Prediction error is below her own usual level (and therefore falling)."""
        rank = self._rank_of_last_window(self._s_avg)
        return rank is not None and rank < SELF_QUANTILE

    def _has_recurring(self) -> bool:
        """She lands in regions she has already been in, more than usual."""
        rank = self._rank_of_last_window(self._recurrence)
        return rank is not None and rank > SELF_QUANTILE

    def _structure_holds(self) -> bool:
        """She stays in a region longer than usual: not pure wandering."""
        rank = self._rank_of_last_window(self._dwell)
        return rank is not None and rank > SELF_QUANTILE

    # ---------- evaluation ----------

    def evaluate(self, t_event: str) -> int:
        """
        Return the current stage (0-4). Observation only; changes nothing.

        Stage 1 needs the first two criteria, stage 2 adds recurrence, stage 3
        adds dwell, stage 4 adds a full observation buffer. All four are
        recorded every time, so a stage that does not advance says which
        criterion is missing (`last_reasons`).

        Stages are cumulative: once a higher stage is reached it is not
        lowered. The buffer condition in stage 4 is a statement about how much
        has been observed, not about her.
        """
        ok_dev = self._dev_not_flat()
        ok_learning = self._learning_visible()
        ok_recurring = self._has_recurring()
        ok_hold = self._structure_holds()
        self.last_reasons = {
            "dev_not_flat": ok_dev,
            "learning_visible": ok_learning,
            "has_recurring": ok_recurring,
            "structure_holds": ok_hold,
            "buffer_full": len(self._dev) >= HIST,
        }

        idx = 0
        if ok_dev and ok_learning:
            idx = 1
        if idx >= 1 and ok_recurring:
            idx = 2
        if idx >= 2 and ok_hold:
            idx = 3
        if idx >= 3 and self.last_reasons["buffer_full"]:
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

    # ---------- persistence ----------
    # 2026-11-16: 阶段之前没有落盘。P3 每次重启都新建 StageMonitor，`_current`
    # 从 0 开始——于是 events.jsonl 里出现 4 -> 0（15:35 mature，16:21 又回到 0）。
    # ARCHITECTURE §8 写的是「阶段转换单向、没有退行机制」，那要求它能跨重启成立。
    #
    # 阶段只被记录、不控制任何层（本文件开头那句），所以补持久化不改变她的行为，
    # 只是让"她走到哪了"这个事实不再随重启归零。

    def to_dict(self) -> dict:
        return {
            "current": int(self._current),
            "history": [[str(t), int(i)] for t, i in self._history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StageMonitor":
        mon = cls()
        mon._current = int(data.get("current", 0))
        for item in data.get("history") or []:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                mon._history.append((str(item[0]), int(item[1])))
        return mon
