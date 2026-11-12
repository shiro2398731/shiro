"""
src/L5_cognition/transitions.py

Attractor transitions.

Definition
----------

A transition occurs when the attractor nearest to the current state,
among the salient ones, changes identity.

The salient set is determined by the depth distribution of the state space
itself (see StateSpace.salient), so the boundary between "which attractor
dominates here" moves with the measured structure. No fixed distance
threshold is used.

Salience vs. assignment
-----------------------

Two different relations are involved and must not be confused:

    assignment (StateSpace.record)  -- used to form and deepen attractors.
        Criterion: normalized distance < 1 characteristic scale.

    nearest (TransitionDetector)    -- used to label the current mental state.
        Criterion: minimum normalized distance among salient attractors.
        There is no cutoff; every state has a nearest attractor.

A transition therefore reports a change in the label of the state, which
does not by itself mean the depth structure changed.

Noise-driven vs. trajectory-driven
----------------------------------

`noise_driven` compares the velocity just before the switch with the
displacement direction:
    - high speed aligned with displacement : the trajectory carried it
    - low speed, or displacement opposed    : fluctuation drove it
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .dynamics import DynamicState
from .terrain import StateSpace


@dataclass
class Transition:
    """A change of the nearest salient attractor."""
    from_attractor: Optional[int]
    to_attractor: int
    t_event: str
    speed_before: float
    displacement: float
    displacement_aligned: float
    noise_driven: bool


class TransitionDetector:
    """
    Reports changes of the nearest salient attractor.

    The frequency of calls is decided by the caller; this class assumes
    nothing about how often it is invoked.
    """

    def __init__(self, state_space: StateSpace):
        self.state_space = state_space
        self._current: Optional[int] = None
        self._last_pos: Optional[np.ndarray] = None
        self.history: list[Transition] = []

    def nearest_salient(self, x: np.ndarray) -> Optional[int]:
        """Index of the nearest salient region -- **index into the attractor table**.

        Same convention as `StateSpace._nearest_normalized` and `graph.touch`.
        Before 2026-11-12 this returned an index into the salient *subset*,
        which is a different number for the same region as soon as one more
        region becomes salient.
        """
        n = self.state_space.nearest_attractor(x)
        return None if n is None else n[0]

    # Compatibility alias.
    def assign(self, x: np.ndarray) -> Optional[int]:
        return self.nearest_salient(x)

    def observe(self, st: DynamicState, t_event: str) -> Optional[Transition]:
        idx = self.nearest_salient(st.x)
        prev = self._current
        prev_pos = self._last_pos
        self._current = idx
        self._last_pos = st.x.copy()

        if idx is None or idx == prev or prev is None:
            return None

        speed_before = float(np.linalg.norm(st.v))
        disp = 0.0
        aligned = 0.0
        if prev_pos is not None:
            d = st.x - prev_pos
            disp = float(np.linalg.norm(d))
            if disp > 1e-12 and speed_before > 1e-12:
                aligned = float((d / disp) @ (st.v / speed_before))

        noise_driven = (aligned < 0.2) or (speed_before < 1e-3)

        ev = Transition(
            from_attractor=prev,
            to_attractor=idx,
            t_event=t_event,
            speed_before=speed_before,
            displacement=disp,
            displacement_aligned=aligned,
            noise_driven=noise_driven,
        )
        self.history.append(ev)
        return ev

    @property
    def current_attractor(self) -> Optional[int]:
        return self._current

    # Compatibility alias.
    @property
    def current_basin(self) -> Optional[int]:
        return self._current

    def recent_sequence(self, k: int = 8) -> list[int]:
        seq: list[int] = []
        if self._current is not None:
            seq.append(self._current)
        for ev in reversed(self.history[-k:]):
            seq.insert(0, ev.from_attractor if ev.from_attractor is not None else -1)
        return seq
