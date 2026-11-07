"""
src/L0_substrate/metabolism.py

Turnover of connections, not learning.

Two things that look similar and are not
----------------------------------------

**Plasticity** changes a structure because of what happened in it. A connection
that carried activity becomes stronger. The structure afterwards is a different
structure, in the sense that matters: what used to be a weak path is now a
strong one. That belongs to a layer that remembers, not to the substrate.

**Metabolism** replaces what wears out so that nothing accumulates. A connection
is removed and a new one takes its place somewhere else, drawn from the same
distribution. The structure afterwards is the same structure: the same number of
connections, the same weights, the same statistics. What changed is which
particular connections exist.

An earlier version of this module did the first thing and was called the second.
It strengthened connections in proportion to activity, which is learning in the
bottom layer, in the one place that is supposed to be the body.

What turnover preserves
-----------------------

The point is not that the structure drifts. The point is that it does not:

    same count       the number of connections is fixed
    same distribution the weights are drawn from the initial distribution
    different identity  which connections exist keeps changing

A system whose substrate is fixed is not a body; it is hardware. A system whose
substrate drifts is not stable; it is a different system each time you look.
Turnover is the arrangement where the substrate is continuously replaced and the
result is continuously the same shape. That is what "still the same thing" means
at the bottom.

What the rate means
-------------------

`turnover` is the fraction of existing connections replaced per step. It is
tiny: at 1e-5 per step, one connection in a hundred thousand is replaced each
step, so the whole substrate turns over in about a hundred thousand steps.

The rate is not required to be constant. `set_rate_scale` multiplies it by a
factor the caller supplies, so turnover can be faster while more is happening
and slower while little is. Nothing here decides what that factor should be.

Why replacement is not paired with removal
------------------------------------------

A removed connection is not replaced at the same position. If it were, nothing
would turn over; the old one would simply be renewed in place, which is the same
as no turnover at all. The replacement is drawn from the whole mask, so a
connection that disappears is gone and a different one appears elsewhere.

Over the short term this is indistinguishable from no change at all, which is
the intended property. Over a long time the substrate has been entirely renewed
while remaining the same substrate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Turnover:
    """
    Continuous replacement of connections in the coupling matrices.

    turnover : fraction of the existing connections replaced per step
    seed     : turnover is random; a fixed seed makes a run reproducible
    """

    def __init__(self, turnover: float = 1e-5, seed: int = 6151):
        self.turnover = float(turnover)
        self.rng = np.random.default_rng(seed)
        self._rate_scale: float = 1.0
        self._replaced: dict[str, int] = {}
        self._steps: int = 0
        # Replacement weights are drawn from the distribution the structure
        # started with, not from its current state. Drawing from the current
        # state makes any drift reinforce itself: once the mean moves a little,
        # replacements are more likely to carry it further. Over 200000 steps
        # that took mid_to_slow from 0.0249 to 0.0213, a change nobody asked
        # for. The initial distribution cannot drift because it does not change.
        self._base: dict[str, np.ndarray] = {}

    def attach(self, matrices: dict[str, np.ndarray]) -> None:
        """
        Record the weights to draw replacements from.

        Call once, after construction, before the first step.
        """
        self._base = {
            k: np.asarray(v, dtype=float).ravel()[np.asarray(v).ravel() != 0.0].copy()
            for k, v in matrices.items()
        }

    def set_rate_scale(self, scale: Optional[float]) -> None:
        """
        How fast turnover runs, in units of the base rate.

        The caller supplies this. It is meant to follow activity, so that more
        is replaced while more is happening. None restores the base rate.
        """
        if scale is None:
            self._rate_scale = 1.0
            return
        s = float(scale)
        self._rate_scale = 1.0 if not np.isfinite(s) else max(s, 0.0)

    def step(self, matrices: dict[str, np.ndarray]) -> None:
        """
        Replace a small fraction of the existing connections, in place.

        matrices are owned by the caller; they are modified in place because
        copying them every step would dominate the cost.

        For each matrix:
            pick n = turnover * rate_scale * nnz existing connections
            set those to zero
            pick the same number of zero positions
            give them weights drawn from the distribution of the weights that
            were there before the removal
        """
        rate = self._rate_scale
        if rate <= 0.0 or self.turnover <= 0.0:
            return
        self._steps += 1

        frac = self.turnover * rate

        for name, m in matrices.items():
            alive = np.flatnonzero(m.ravel() != 0.0)
            if alive.size == 0:
                continue
            n = int(alive.size * frac)
            # At very small rates most steps replace nothing. Deciding by a
            # draw rather than rounding keeps the expected rate exact instead
            # of quantising it to "one connection every few steps".
            if n == 0:
                if self.rng.random() >= alive.size * frac:
                    continue
                n = 1
            n = min(n, alive.size)

            # Draw the replacement weights from the distribution the matrix
            # started with, so turnover renews connections without moving the
            # distribution that defines what kind of structure this is.
            pool = self._base.get(name)
            if pool is None or pool.size == 0:
                pool = m.ravel()[alive]
            new_w = self.rng.choice(pool, size=n, replace=True)

            drop = self.rng.choice(alive, size=n, replace=False)
            m.ravel()[drop] = 0.0

            # Positions that are now empty, excluding the ones just dropped so a
            # connection cannot be renewed in place.
            empty = np.flatnonzero(m.ravel() == 0.0)
            if empty.size < n:
                # Nothing better to do than leave them empty; the count drifts
                # down by a few. Recording it is enough.
                self._replaced[name] = self._replaced.get(name, 0) + drop.size
                continue
            fill = self.rng.choice(empty, size=n, replace=False)
            m.ravel()[fill] = new_w

            self._replaced[name] = self._replaced.get(name, 0) + n

    def describe(self) -> dict:
        return {
            "turnover": self.turnover,
            "rate_scale": self._rate_scale,
            "steps": self._steps,
            "replaced": dict(self._replaced),
        }

    # ---------- persistence ----------

    def to_dict(self) -> dict:
        return {
            "turnover": self.turnover,
            "rate_scale": self._rate_scale,
            "replaced": dict(self._replaced),
            "steps": self._steps,
            "base": {k: v.tolist() for k, v in self._base.items()},
            "rng": self.rng.bit_generator.state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Turnover":
        t = cls(turnover=float(d.get("turnover", 1e-5)))
        t._rate_scale = float(d.get("rate_scale", 1.0))
        t._replaced = {str(k): int(v) for k, v in (d.get("replaced") or {}).items()}
        t._steps = int(d.get("steps", 0))
        t._base = {k: np.asarray(v, dtype=float)
                   for k, v in (d.get("base") or {}).items()}
        st = d.get("rng")
        if st:
            try:
                t.rng.bit_generator.state = st
            except Exception:
                pass
        return t
