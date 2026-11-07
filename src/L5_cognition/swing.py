"""
src/L5_cognition/swing.py

A global slow variable, so the trajectory sweeps instead of settling.

Why per-region rest was not enough
----------------------------------

The first attempt gave every region its own slow scalar. A region that the state
sat in lost its hold, and the state was free to leave. It did not leave.

Releasing a region only removes what was keeping the state there. It does not
create anything that pulls it elsewhere. With no preference among the remaining
regions, a freed state sits where it is.

What is needed is not release but direction: something that makes one part of
the space more reachable than another, and then reverses.

The structure taken from L0
---------------------------

L0 moves because a single slow variable w acts on the fast system and reverses
it. The corresponding thing here is a single global vector:

    ds/dt = +slow_rate * (target - s)
    target = a direction derived from where the state just was

with s a vector in the same 128-dimensional space as the state. The potential
gains a term that tilts it along s. Because s is one shared quantity, the whole
landscape tilts together: the space does not merely become available in a
different place, it acquires a preferred direction that the state moves along.

When the state has moved far enough in that direction, target is recomputed from
the new position and s reverses. That is the sweep.

Where the target comes from
---------------------------

Not from a constant and not from a designer.

target is built from the recent displacement of the state, reflected through
the point the state started from when s last turned: s is meant to push the
state back toward where it has been, so the motion alternates instead of
drifting away.

The magnitude of the tilt is not fixed either. It is the median gradient
magnitude the structure has actually produced, so the tilt is comparable to the
structure's own pull rather than being stronger or weaker by an arbitrary
factor.

What this is not
----------------

It is not a second potential. s enters as a linear term, exactly like the drive
and deflection terms already in Landscape; it changes where the space slopes,
not where its minima are.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class Swing:
    """
    Global slow direction that keeps the trajectory in motion.

    slow_rate : how fast the direction itself follows its target
    turn_frac : fraction of the travelled distance at which it turns around
    window    : how many recent displacements define the direction
    """

    def __init__(
        self,
        dim: int = 128,
        slow_rate: float = 0.01,
        turn_frac: float = 0.5,
        window: int = 32,
    ):
        self.D = int(dim)
        self.slow_rate = float(slow_rate)
        self.turn_frac = float(turn_frac)
        self.window = int(window)

        self._s = np.zeros(self.D)          # the slow direction itself
        self._anchor = None                 # position when it last turned
        self._travel = deque(maxlen=self.window)
        self._grad = deque(maxlen=256)
        self._anchor_dist: Optional[float] = None
        self._turns = 0

    # ---------- observation ----------

    def observe_gradient(self, grad_norm: float) -> None:
        if np.isfinite(grad_norm):
            self._grad.append(float(grad_norm))

    def _scale(self) -> float:
        """Tilt magnitude, in the units the structure itself uses."""
        if not self._grad:
            return 0.0
        return float(np.median(np.asarray(self._grad, dtype=float)))

    # ---------- advance ----------

    def step(self, x: np.ndarray) -> None:
        """
        Advance the slow direction by one step.

        x: current state position.
        """
        x = np.asarray(x, dtype=float)

        if self._anchor is None:
            self._anchor = x.copy()
            self._travel.clear()
            return

        disp = x - self._anchor
        dist = float(np.linalg.norm(disp))
        self._travel.append(disp.copy())

        if self._anchor_dist is None:
            self._anchor_dist = dist

        # Turn when the state has moved away from the anchor by more than the
        # amount it had already covered when the anchor was set. The threshold
        # is the state's own previous excursion, not a fixed distance.
        if dist > 0 and self._anchor_dist > 0 and dist > self._anchor_dist * (1.0 + self.turn_frac):
            self._anchor = x.copy()
            self._travel.clear()
            self._anchor_dist = dist
            self._turns += 1
            # Reversing is implicit: the target is built from displacements
            # measured from the new anchor, which points the other way.
            return

        # Target: pull back toward where the state has been, so the sweep
        # alternates. Mean over the window, negated.
        if len(self._travel) >= 4:
            mean_disp = np.mean(np.stack(list(self._travel)), axis=0)
            n = float(np.linalg.norm(mean_disp))
            target = -(mean_disp / n) if n > 1e-12 else np.zeros(self.D)
        else:
            target = np.zeros(self.D)

        self._s = self._s + self.slow_rate * (target - self._s)

    # ---------- readout ----------

    def tilt(self) -> Optional[np.ndarray]:
        """
        The direction to add to the potential's slope, already scaled.

        None before the direction has any magnitude, so a run that has not
        turned yet is not tilted by an arbitrary zero vector.
        """
        m = float(np.linalg.norm(self._s))
        if m <= 1e-12:
            return None
        s = self._scale()
        if s <= 0:
            return None
        return (self._s / m) * s

    @property
    def turns(self) -> int:
        return self._turns

    def summary(self) -> dict:
        return {
            "s_norm": float(np.linalg.norm(self._s)),
            "scale": self._scale(),
            "turns": self._turns,
            "travel": len(self._travel),
        }

    # ---------- persistence ----------

    def to_dict(self) -> dict:
        return {
            "s": [float(v) for v in self._s],
            "anchor": None if self._anchor is None else [float(v) for v in self._anchor],
            "anchor_dist": self._anchor_dist,
            "turns": self._turns,
            "slow_rate": self.slow_rate,
            "turn_frac": self.turn_frac,
            "window": self.window,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Swing":
        s = np.asarray(d.get("s") or [], dtype=float)
        sw = cls(
            dim=int(s.size) if s.size else 128,
            slow_rate=float(d.get("slow_rate", 0.01)),
            turn_frac=float(d.get("turn_frac", 0.5)),
            window=int(d.get("window", 32)),
        )
        if s.size:
            sw._s = s
        a = d.get("anchor")
        sw._anchor = np.asarray(a, dtype=float) if a else None
        sw._anchor_dist = d.get("anchor_dist")
        sw._turns = int(d.get("turns", 0))
        return sw
