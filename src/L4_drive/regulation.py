"""
src/L4_drive/regulation.py

Regulation: what the system does when part of its structure is draining.

Where this comes from
---------------------

Viability (see viability.py) reports which regions sit below the level their
own gain and loss rates can sustain. Di Paolo's chain says the next step is
*regulation*: the system's activity changes the distance between where it is and
where it can continue.

That is the whole of it. No goal is represented, no reward is computed, no
drive is named. The mechanism is:

    regions that are draining attract the state less
    regions that are sustaining attract the state more

Why this is not a value judgement
---------------------------------

The system does not decide that one region is better than another. It observes
that a region is losing depth faster than it regains it, and lets the dynamics
follow that fact. A region with no depth cannot pull anything; this only makes
the loss visible to the trajectory earlier than the depth would.

Why the pull changes rather than the depth
------------------------------------------

Lowering the depth of a draining region would simply be the recovery rate
applied harder, which is already happening. What does not yet happen is that
the *trajectory* responds to the loss. So the regulation acts on how strongly a
region pulls, not on how deep it is.

What "pull" is here
-------------------

    gradient contribution = amp * exp(-d^2 / (2 sigma^2)) * (x - mu) / sigma^2

Multiplying `amp` by a factor per region is exact and cheap. The factor is
derived from viability ratio, not chosen:

    factor = ratio ** beta      where ratio = depth / carrying

    ratio > 1 : region above its level, factor > 1, pulls harder
    ratio < 1 : region below its level, factor < 1, pulls weaker

beta controls how strongly viability modulates the pull. It is an engineering
scale, not a judgement: it does not decide which regions matter, only how much
the measured viability is allowed to speak.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.L4_drive.viability import Viability


class Regulation:
    """
    Turns viability measurements into per-region pull factors.
    """

    def __init__(self, beta: float = 1.0, floor: float = 0.0):
        """
        beta  : exponent linking viability ratio to pull strength
        floor : lower bound on the factor. Zero means a fully drained region
                stops pulling entirely, which is what the measurement says.

        There is no ceiling. It was there because the viability ratio was being
        computed wrongly: the visit rate used the whole run as its denominator,
        so a young region looked almost never visited, its carrying depth was
        understated, and its ratio came out twenty times too high. A cap on the
        factor hid that.

        With the ratio measured over the region's own lifetime, the population
        maximum is about 1.35. Sweeping the cap over 2.0, 4.0, 8.0 and none at
        all produced the same attractor count, the same edge count, the same
        average speed and the same transition count to within noise, because no
        factor ever reached the cap. A bound that never binds is not a bound; it
        is a leftover.

        On beta
        ------
        beta = 1 makes the pull equal to the viability ratio: a region twice
        past what it can carry pulls twice as hard. That is the identity
        relation, and it is chosen because it is the only value that adds no
        assumption of its own.
        """
        self.beta = float(beta)
        self.floor = float(floor)
        self.viability = Viability()
        self._last: Optional[np.ndarray] = None

    def factors(self, state_space) -> np.ndarray:
        """
        Pull factor per attractor, in attractor order.
        """
        regions = self.viability.measure(state_space)
        if not regions:
            self._last = np.zeros(0)
            return self._last

        n = len(regions)
        f = np.ones(n, dtype=float)
        for r in regions:
            ratio = max(r.ratio, 0.0)
            f[r.index] = ratio ** self.beta
        np.maximum(f, self.floor, out=f)
        self._last = f
        return f

    @property
    def last(self) -> Optional[np.ndarray]:
        return None if self._last is None else self._last.copy()

    def summary(self, state_space) -> dict:
        before = self.viability.summary(state_space)
        f = self.factors(state_space)
        if f.size:
            before["factor_median"] = float(np.median(f))
            before["factor_min"] = float(f.min())
            before["factor_max"] = float(f.max())
            before["below_0_5"] = int((f < 0.5).sum())
        return before
