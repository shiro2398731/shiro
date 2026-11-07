"""
src/L5_cognition/rest.py

A slow variable per region, so the trajectory keeps moving on its own.

The problem
-----------

Without noise, an attractor landscape is a settling device:

    dv = -gamma*v - grad V(x)

The state descends and stops. The way out used so far was to add a fluctuation
term from outside. That works, but it makes L5 reactive: movement exists only
because something external pushes. Stop pushing, and it stops.

A system that keeps going has to carry its own cause of motion. L0 does: the
recovery variable w makes the fast variable unable to stay put. What is missing
in L5 is the same kind of term.

What is added
-------------

Each region carries a depth-like value u that falls while the region is
occupied and recovers otherwise. The region's actual depth is the structure's
own depth minus u, so a region that has been occupied for a while loses its
hold, and somewhere else becomes reachable.

    u = u + fatigue_rate * (1 - u)      while the region IS occupied
    u = u + rest_rate * (u_target - u)  while it is NOT occupied

`u` is bounded to [0, 1]. It is an amount of release, not a correction to the
structure's own depth: the region's effective depth is depth * (1 - u), so a
region that has been occupied for a while loses its hold, and somewhere else
becomes reachable. Rest can make a region less attractive, never more.

Where u recovers to
-------------------

Not to a constant.

If u recovered to a fixed value, the system would have an original shape and
every excursion would be erased. That is the property that was rejected: it
would mean nothing accumulated. L0 does recover to a constant, but L0 has a
single point and no history to speak of.

Here u_target is the mean of u over the last `window` recoveries. The window
rolls, so the target moves with the region's recent past and never settles.
The motion is bounded because every step pulls u toward that mean, and it never
stops because the mean itself keeps changing.

This is the same structure the record of personality change describes: a level
that shifts with experience and is not pulled back to where it started.

What this class does not do
---------------------------

It does not decide what u means. It is a slow scalar attached to a region,
nothing more. Naming it would be naming a state that belongs to the system.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

import numpy as np


class RestField:
    """
    Per-region slow variable.

    fatigue_rate : how fast occupation drains a region's hold
    rest_rate    : how fast an unoccupied region recovers
    window       : how many past values define the recovery target
    """

    def __init__(
        self,
        fatigue_rate: float = 0.02,
        rest_rate: float = 0.002,
        window: int = 64,
    ):
        self.fatigue_rate = float(fatigue_rate)
        self.rest_rate = float(rest_rate)
        self.window = int(window)
        # Dense arrays over region index, grown on demand. Iterating a dict and
        # converting deques to arrays every step cost more than the whole
        # potential evaluation once the region count reached several hundred.
        self._u = np.zeros(0, dtype=float)
        self._sum = np.zeros(0, dtype=float)     # running sum of history
        self._hist = []                          # list of deques, per region
        self._seen = np.zeros(0, dtype=bool)

    def _grow(self, n: int) -> None:
        if n <= self._u.size:
            return
        m = max(n, self._u.size * 2, 16)
        u = np.zeros(m, dtype=float)
        s = np.zeros(m, dtype=float)
        seen = np.zeros(m, dtype=bool)
        u[:self._u.size] = self._u
        s[:self._sum.size] = self._sum
        seen[:self._seen.size] = self._seen
        for i in range(self._u.size, m):
            self._hist.append(deque(maxlen=self.window))
        self._u, self._sum, self._seen = u, s, seen

    def value(self, idx: int) -> float:
        return float(self._u[idx]) if idx < self._u.size else 0.0

    def target(self, idx: int) -> float:
        """Recovery target: the region's own recent level, not a constant."""
        if idx >= self._hist.__len__():
            return 0.0
        h = self._hist[idx]
        if not h:
            return 0.0
        return float(self._sum[idx] / len(h))

    # ---------- advance ----------

    def step(self, occupied: Optional[int], rest_rate_scale: float = 1.0) -> None:
        """
        Advance every tracked region by one step.

        occupied: index of the region the state is in, or None.
        rest_rate_scale: supplied by upstream if the recovery rate is to vary;
                         left at 1.0 nothing outside influences it.

        u rises while occupied and falls while not. Rising is what makes the
        region release its hold; it has to start from zero and climb, because
        a value that starts at zero and is only ever decreased stays at zero
        and nothing moves.

        Only tracked regions advance. A region the state has never entered has
        no history to recover toward, and creating entries for all of them made
        this loop the dominant cost of a step.
        """
        n = self._seen.size
        if n == 0:
            if occupied is None:
                return
            self.touch(occupied)
            n = self._seen.size

        act = np.flatnonzero(self._seen)
        if act.size == 0:
            if occupied is not None:
                self.touch(occupied)
            return

        u = self._u[act]
        cnt = np.array([len(self._hist[i]) for i in act], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            tgt = np.where(cnt > 0, self._sum[act] / np.maximum(cnt, 1.0), 0.0)

        is_occ = np.zeros(act.size, dtype=bool)
        if occupied is not None:
            pos = np.searchsorted(act, occupied)
            if pos < act.size and act[pos] == occupied:
                is_occ[pos] = True

        u_new = np.where(
            is_occ,
            u + self.fatigue_rate * (1.0 - u),
            u + self.rest_rate * rest_rate_scale * (tgt - u),
        )
        u_new = np.clip(u_new, 0.0, 1.0)

        self._u[act] = u_new
        for j, i in enumerate(act):
            h = self._hist[i]
            if len(h) == h.maxlen:
                self._sum[i] -= h[0]
            h.append(float(u_new[j]))
            self._sum[i] += float(u_new[j])

    def touch(self, idx: int) -> None:
        """Register a region so it starts being tracked."""
        self._grow(idx + 1)
        self._seen[idx] = True

    def prune(self, alive: set) -> None:
        n = self._u.size
        if n == 0:
            return
        keep = np.array([i in alive for i in range(n)], dtype=bool)
        self._u = np.where(keep, self._u, 0.0)
        self._seen = keep

    def remap(self, mapping: Dict[int, int]) -> None:
        """Renumber after the state space merges or removes regions."""
        n = self._u.size
        new_u = np.zeros(n, dtype=float)
        new_seen = np.zeros(n, dtype=bool)
        for old, new in mapping.items():
            if old < n and new < n:
                new_u[new] = self._u[old]
                new_seen[new] = self._seen[old]
        self._u = new_u
        self._seen = new_seen

    # ---------- observation ----------

    def summary(self) -> dict:
        act = self._seen
        if not act.any():
            return {"n": 0, "u_mean": 0.0, "u_max": 0.0, "u_min": 0.0}
        vals = self._u[act]
        return {
            "n": int(vals.size),
            "u_mean": float(vals.mean()),
            "u_max": float(vals.max()),
            "u_min": float(vals.min()),
        }

    # ---------- persistence ----------

    def to_dict(self) -> dict:
        act = np.flatnonzero(self._seen)
        return {
            "idx": [int(i) for i in act],
            "u": [float(self._u[i]) for i in act],
            "hist": [[float(x) for x in self._hist[i]] for i in act],
            "fatigue_rate": self.fatigue_rate,
            "rest_rate": self.rest_rate,
            "window": self.window,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RestField":
        f = cls(
            fatigue_rate=float(d.get("fatigue_rate", 0.02)),
            rest_rate=float(d.get("rest_rate", 0.002)),
            window=int(d.get("window", 64)),
        )
        idxs = d.get("idx") or []
        vals = d.get("u") or []
        hists = d.get("hist") or []
        for j, i in enumerate(idxs):
            idx = int(i)
            f.touch(idx)
            f._u[idx] = float(vals[j]) if j < len(vals) else 0.0
            h = f._hist[idx]
            if j < len(hists):
                for x in hists[j]:
                    h.append(float(x))
            f._sum[idx] = float(sum(h))
        return f
