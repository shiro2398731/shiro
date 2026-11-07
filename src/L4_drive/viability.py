"""
src/L4_drive/viability.py

Viability: whether the structure can keep itself going.

Why this exists
---------------

Dynamical systems that are only damped will settle and stop. shiro's state
equation has three terms -- friction, attraction, fluctuation -- and none of
them expresses "this region can no longer support itself". A structure that is
being eroded at the same rate it is being reinforced is indistinguishable, in
that equation, from one that is thriving.

Di Paolo (2005, "Autopoiesis, Adaptivity, Teleology, Agency") argues that
self-production alone is not enough for a system to have a point of view. What
is required is *adaptivity*: the capacity to regulate the system's relation to
its own conditions of viability. The chain he proposes is

    autopoiesis + adaptivity -> sense-making -> agency

Sense-making does not require a goal, a reward, or a representation of what is
good. It requires only that some states lie inside the region where the system
can continue and others outside, and that the system be affected by the
difference.

What viability is here
----------------------

Not a configured number. It is measured from the structure over time.

For each attractor, two rates compete:

    gain rate   -- how often states are assigned into it
    loss rate   -- how fast its depth is being reduced

A region whose loss outruns its gain is being erased. A region where gain keeps
up is being sustained. The ratio of the two is what "can this keep going" means
here, expressed in the system's own units.

Nothing in this module decides that a region *should* be kept. It only reports
which regions are holding and which are draining. What uses that report is
elsewhere.

Measurement
-----------

Depth changes by two processes that run at different rates:

    assignment  -- happens only when a state falls inside this region
    recovery    -- happens on a fixed cadence, for every region

The earlier version of this module assumed assignment happened every step,
which produced a carrying level of ~0.998 for almost every region and a
"below carrying" fraction of 99.5%. That number meant nothing: it said only
that regions are not full, which is always true.

What is needed is the rate at which a region is actually reached, not the rate
at which it would be reached if it were reached constantly.

Let

    p  = assignments per step, measured as assignments / observations
         for this region (both counted since birth)
    R  = recovery applied per step = recover_rate * decay / recover_every

then over one step, in expectation,

    gain = p * assign_ratio * (ceiling - amp)
    loss = R * amp

balancing these gives

    amp* = ceiling * (p * a) / (p * a + R)

where a = assign_ratio. When p is small, amp* is small: a region that is
rarely reached cannot hold much depth. When p is high, amp* approaches the
ceiling.

The viability ratio is amp / amp*, so 1.0 means this region holds exactly as
much depth as its own visit rate supports. Below 1.0 means it holds less than
it could; above 1.0 means it is holding more than its current visit rate earns,
which is what a recently-formed deep region looks like before it fades.

All quantities come from StateSpace's own parameters and from observed counts.
No constant is chosen by the designer beyond the numerical tolerance used to
avoid division by zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Keep in step with terrain.MIN_RATE_SCALE. A region whose return interval is
# very long still fades, just slowly; without a floor a single stray region can
# end up effectively frozen.
_MIN_RATE_SCALE = 0.01
_MIN_ASSIGN_FOR_INTERVAL = 8


def _interval_scale(state_space, attractor) -> float:
    """
    Slowdown factor StateSpace.recover applies to this region's decay.

    Mirrors the rule in terrain.recover: a region whose own revisit interval is
    longer than the recovery half-life decays in proportion to how rarely it is
    reached, floored so that nothing is frozen entirely. Kept as a separate
    function here because viability must use the same number the state space
    actually uses; computing a sustaining depth under a different decay rate
    than the one in force gives a ratio that measures the disagreement, not the
    system.

    If the state space does not expose what this needs, the scale is 1.0 and the
    base rate is used, which is the conservative choice: it understates the
    sustaining depth, so the ratio leans high rather than low.
    """
    try:
        total = max(int(getattr(state_space, "_total_steps", 0)), 1)
        assigned = int(getattr(attractor, "_assigned", 0))
        rate = float(getattr(state_space, "recover_rate", 0.0))
        every = max(int(getattr(state_space, "recover_every",
                                getattr(state_space, "_recover_every_hint", 16))), 1)
        if assigned < _MIN_ASSIGN_FOR_INTERVAL or rate <= 0 or rate >= 1:
            return 1.0
        half = float(np.log(0.5) / np.log(max(1.0 - rate, 1e-12))) * every
        interval = total / float(assigned)
        if interval <= half or half <= 0:
            return 1.0
        return max(half / interval, _MIN_RATE_SCALE)
    except Exception:
        return 1.0


@dataclass
class Region:
    """
    How one attractor stands relative to what it can sustain.

    index       position in the state space's attractor list
    depth       current depth
    carrying    the depth this region's own gain and loss rates balance at
    ratio       depth / carrying, so above 1 means above its own level
    drain       how much of the depth per step is being lost, relative to gain
    """
    index: int
    depth: float
    carrying: float
    ratio: float
    drain: float

    def describe(self) -> str:
        return (f"[{self.index}] depth {self.depth:.4f} / carrying "
                f"{self.carrying:.4f}  ratio {self.ratio:.2f}")


class Viability:
    """
    Reports, for each region, whether it is above or below the level it can
    sustain.

    Read-only with respect to the state space.
    """

    def __init__(self, tolerance: float = 1e-12):
        """
        tolerance : numerical guard only, for divisions. Not a threshold on
                    what counts as viable.
        """
        self.tol = float(tolerance)
        self._last: list[Region] = []

    # ---------- carrying level ----------

    @staticmethod
    def carrying_depth(state_space, attractor, visit_rate: float) -> float:
        """
        The depth this region can sustain given how often it is actually
        reached.

            amp* = ceiling * (p * a) / (p * a + R_eff)

        where
            p     = visit_rate (assignments per step, observed)
            a     = assign_ratio (state space parameter)
            R_eff = the recovery this region actually experiences per step

        R_eff is not recover_rate / recover_every on its own. StateSpace.recover
        slows a region's decay when its revisit interval is longer than the
        recovery half-life, so a rarely visited region recovers more slowly than
        the base rate suggests. Using the base rate here while recover() uses
        the scaled one made the two disagree by an order of magnitude: the
        measured ratio of depth to carrying depth had a median of 4.3 and a
        maximum of 38, which is not a property of the system but of two numbers
        computed under different assumptions.

        See the module docstring for the derivation.
        """
        a = state_space.assign_ratio
        f = attractor._decay_factor if attractor._decay_factor is not None else 1.0
        # The state space stores this as _recover_every_hint; P2 also sets a
        # plain recover_every. Both are checked so that neither name silently
        # falls back to the default.
        every = max(int(getattr(state_space, "recover_every",
                                getattr(state_space, "_recover_every_hint", 16))), 1)
        rate_scale = _interval_scale(state_space, attractor)
        R = state_space.recover_rate * f * rate_scale / every
        num = max(visit_rate, 0.0) * a
        denom = num + R
        if denom <= 0:
            return 0.0
        return float(state_space.ceiling * num / denom)

    # ---------- measure ----------

    def measure(self, state_space) -> list[Region]:
        """
        Measure every attractor. Sorted by ratio ascending, so the ones furthest
        below their own carrying level come first.

        Visit rate is assignments / steps since the region was created. Steps
        since creation are not stored per attractor, so the step counter of the
        state space is used as an upper bound; for a region created recently
        this underestimates the rate, which biases the carrying level downward
        and therefore makes a new region look better sustained than it is. That
        bias is accepted rather than fixed by adding a per-region creation
        counter to the checkpoint format, which would be a larger change.
        """
        attrs = state_space.attractors()
        if not attrs:
            self._last = []
            return []

        total_steps = max(int(getattr(state_space, "_total_steps", 0)), 1)

        out: list[Region] = []
        for i, a in enumerate(attrs):
            # assignments observed for this region, against the steps that have
            # passed since it was created. Dividing by the whole run makes every
            # young region look almost never visited, which understates what it
            # can carry and inflates its ratio.
            assigned = float(getattr(a, "_assigned", 0)) or float(len(a.members))
            born = int(getattr(a, "_born_step", 0))
            life = max(total_steps - born, 1)
            p = assigned / life

            car = self.carrying_depth(state_space, a, p)
            ratio = a.amp / car if car > self.tol else 0.0

            r = state_space.assign_ratio
            headroom = max(state_space.ceiling - a.amp, 0.0)
            gain = p * r * headroom
            f = a._decay_factor if a._decay_factor is not None else 1.0
            every = max(int(getattr(state_space, "recover_every",
                                    getattr(state_space, "_recover_every_hint", 16))), 1)
            loss = a.amp * state_space.recover_rate * f / every
            drain = loss / gain if gain > self.tol else float("inf")

            out.append(Region(
                index=i,
                depth=float(a.amp),
                carrying=car,
                ratio=float(ratio),
                drain=float(drain),
            ))

        out.sort(key=lambda x: x.ratio)
        self._last = out
        return out

    # ---------- aggregate ----------

    def summary(self, state_space) -> dict:
        """
        One number for how much of the structure is below its own carrying
        level.

        Not a score and not a target: it is the fraction of regions that hold
        less depth than their own visit rate would support.
        """
        rs = self.measure(state_space)
        if not rs:
            return {"n": 0, "below": 0, "frac_below": 0.0,
                    "min_ratio": 0.0, "median_ratio": 0.0}

        ratios = np.array([r.ratio for r in rs], dtype=float)
        below = int((ratios < 1.0).sum())
        return {
            "n": len(rs),
            "below": below,
            "frac_below": float(below / len(rs)),
            "min_ratio": float(ratios.min()),
            "median_ratio": float(np.median(ratios)),
        }

    def wall(self, state_space) -> Optional[float]:
        """
        The potential level beyond which the space stops being traversable.

        Definition follows the ecological sense of resilience rather than of
        stability (Holling 1973; Ludwig, Walker & Holling 1997):

            stability  -- returning to an equilibrium after disturbance
            resilience -- absorbing disturbance without leaving the state
                          regime the system occupies

        A state space whose structure is all shallow has no stable
        equilibria of any depth, yet it can still be resilient if the entire
        depth distribution is uniform. Conversely a single very deep region
        can coexist with a shallow background without the background being
        "failing". An absolute depth cut-off cannot distinguish these cases,
        and the earlier version of this method, which used the median depth,
        made nearly the whole space untraversable whenever the structure as a
        whole was shallow (measured fraction below the median: 0.97).

        What is used instead: the median of the depth each point in the space
        would have from the structure, sampled across the observed range.
        That is the typical depth the space actually offers where states go,
        taken as a level rather than as a property of individual regions. It
        still comes from the structure, but it is measured where the state
        goes rather than from the regions themselves.

        Returns None when there is not enough structure to define a level.
        """
        anchors = getattr(state_space, "_recent", None)
        if anchors is None or len(anchors) < 8:
            return None
        pts = np.stack(list(anchors))
        depths = np.array([state_space._depth_at(p) for p in pts], dtype=float)
        depths = depths[depths > 0]
        if depths.size < 8:
            return None
        return -float(np.median(depths))

    @property
    def last(self) -> list[Region]:
        return list(self._last)

    # ---------- which regions are draining ----------

    def draining(self, state_space, count: int = 8) -> list[Region]:
        """The `count` regions furthest below their own carrying level."""
        rs = self.measure(state_space)
        return rs[:count]
