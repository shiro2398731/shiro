"""
src/L5_cognition/workspace.py

Global workspace: which representation becomes globally available.

Model
-----

Global Neuronal Workspace theory (Mashour & Dehaene 2020, Neuron):

    In the conscious state, a nonlinear network ignition amplifies and
    sustains a representation, making it globally available to processing
    modules. The global availability is what is subjectively experienced.

Mechanism: threshold dynamics over competing representations, not scoring.

Implementation
--------------

The workspace reads the state space and reports which attractor is
currently dominant. Competitors are the salient attractors; each has an
effective strength

    strength = amp * exp(-||x - mu||^2 / (2 sigma^2))

The reference level is the median of previously observed peak strengths:
an attractor must exceed what this system has typically produced to count
as ignited. The threshold therefore shifts with the system's own history.

Sustaining
----------

After ignition, the representation stays active while the state remains
within one characteristic scale of the ignited attractor's centroid, and
ends when the state leaves. Duration is thus a consequence of the
dynamics, not a configured number.

The workspace does not modify the state space, the potential, or the
trajectory. It only reads.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional

# Where in her own history of peak strengths the bar sits.
#
# It used to be the median, and a median split fires half the time by
# construction -- measured 2026-11-13 (tools/m4_audit.py): 0.4975 ignitions per
# step, and the ignited set was *wider* than the set she actually dwelt in
# (H_ign 4.69 bits vs H_visit 3.79). That is a wide broadcast, not a bottleneck.
#
# A quantile of her own peak history is the same kind of number as the fold
# threshold: nothing is chosen except which quantile, and the value comes from
# her data. 0.90 means "the top tenth of what she has produced".
IGNITION_QUANTILE = 0.90

# How many past ignitions are kept. Buffer discipline (C1), not a criterion.
# Before 2026-11-13 `history` was an unbounded list: at two ignitions per second
# that is ~170k objects per day in a process meant to run for weeks.
HISTORY_CAP = 1024

import numpy as np


@dataclass
class Ignition:
    """One ignition event.

    `attractor_id` is the region's name (uid), not its row: a report that
    outlives the structure it was made in has to keep meaning the same region.
    """
    attractor_id: int
    strength: float
    x: np.ndarray
    t_event: str


class GlobalWorkspace:
    """
    Reports which attractor becomes globally available.
    Reads the state space; does not modify it.
    """

    def __init__(self, history_len: int = 256):
        """
        history_len : number of past peak strengths kept for the reference level.
                      An observation parameter, not a mechanism.
        """
        self._strength_hist: collections.deque = collections.deque(maxlen=history_len)
        self._current: Optional[Ignition] = None
        self.history: collections.deque = collections.deque(maxlen=HISTORY_CAP)

        self._active_mu: Optional[np.ndarray] = None
        self._active_sigma: float = 0.0
        self._active_id: Optional[int] = None

    # ---------- strength ----------

    @staticmethod
    def _strength(attractor, x: np.ndarray) -> float:
        """amp * proximity, proximity = exp(-d^2 / (2 sigma^2))."""
        d = np.asarray(x, dtype=float) - attractor.mu
        d2 = float(d @ d)
        s = attractor.sigma
        if s <= 0:
            return 0.0
        proximity = float(np.exp(-0.5 * d2 / (s * s)))
        return float(attractor.amp) * proximity
    # ---------- main loop ----------

    def observe(self, state_space, x: np.ndarray, t_event: str) -> Optional[Ignition]:
        """
        Observe one state. Returns a new ignition if one occurred, else None.

        The caller must not use the return value to modify cognition.
        """
        x = np.asarray(x, dtype=float)

        # Sustaining: is the state still within one scale of the ignited centroid?
        if self._active_mu is not None and self._active_sigma > 0:
            d = float(np.linalg.norm(x - self._active_mu))
            if d <= self._active_sigma:
                return None
            self._active_mu = None
            self._active_sigma = 0.0
            self._active_id = None

        attrs = state_space.attractors()
        if not attrs:
            return None

        # Reuse the state space's cached centroid and scale matrices instead of
        # stacking them again: this runs every step and building two (n, D)
        # arrays per step is work the state space has already done.
        try:
            state_space._centroid_matrix()
            mus = state_space._mu_cache
            sig = np.maximum(state_space._sigma_vec, np.finfo(float).eps)
        except AttributeError:
            mus = np.stack([a.mu for a in attrs])
            sig = np.maximum(
                np.array([a.sigma for a in attrs], dtype=float),
                np.finfo(float).eps,
            )
        # Depth changes every step, so it is read fresh.
        amp = np.array([a.amp for a in attrs], dtype=float)
        d = mus - x
        d2 = np.einsum("ij,ij->i", d, d)
        strengths = amp * np.exp(-0.5 * d2 / (sig * sig))
        best_i = int(np.argmax(strengths))
        best = float(strengths[best_i])

        if self._strength_hist:
            ref = float(np.quantile(np.asarray(self._strength_hist, dtype=float),
                                    IGNITION_QUANTILE))
        else:
            ref = None

        self._strength_hist.append(best)

        if ref is None:
            return None

        if best <= ref or best <= 1e-12:
            return None

        a = attrs[best_i]
        self._active_id = int(a.uid)
        self._active_mu = a.mu.copy()
        self._active_sigma = float(a.sigma)

        ign = Ignition(
            attractor_id=int(a.uid),
            strength=best,
            x=x.copy(),
            t_event=t_event,
        )
        self._current = ign
        self.history.append(ign)
        return ign

    # ---------- observation ----------

    @property
    def current(self) -> Optional[Ignition]:
        if self._active_mu is None:
            return None
        return self._current

    @property
    def active(self) -> bool:
        return self._active_mu is not None

    # Compatibility alias.
    @property
    def burning(self) -> bool:
        return self._active_mu is not None

    def reference_level(self) -> Optional[float]:
        """The bar a peak has to clear: IGNITION_QUANTILE of her own peaks.

        A quantile rather than the median: the median fires half the time by
        construction, which cannot be a bottleneck (measured, see the constant).
        """
        if not self._strength_hist:
            return None
        return float(np.quantile(np.asarray(self._strength_hist, dtype=float),
                                 IGNITION_QUANTILE))

    # Compatibility alias.
    def baseline(self) -> Optional[float]:
        return self.reference_level()

    @property
    def ignition_count(self) -> int:
        return len(self.history)

    # ---------- persistence ----------
    #
    # Added 2026-11-24: none of this was in the checkpoint, so a restart began
    # with an empty strength history. `reference_level()` then returned None and
    # `observe()` could not ignite until 256 fresh peaks had been collected --
    # and every ignition record was lost, which made "what was globally
    # available" unobservable across restarts.
    #
    # The reference level comes from her own peak history, so this is state, not
    # a cache: without it the bar is undefined after a restart.

    def to_dict(self) -> dict:
        def _ign(i: Ignition | None) -> dict | None:
            if i is None:
                return None
            return {
                "attractor_id": int(i.attractor_id),
                "strength": float(i.strength),
                "x": [float(v) for v in i.x],
                "t_event": str(i.t_event),
            }

        return {
            "strength_hist": [float(v) for v in self._strength_hist],
            "history": [_ign(i) for i in self.history],
            "current": _ign(self._current),
            "active_mu": (None if self._active_mu is None
                          else [float(v) for v in self._active_mu]),
            "active_sigma": float(self._active_sigma),
            "active_id": (None if self._active_id is None
                          else int(self._active_id)),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalWorkspace":
        g = cls()
        for v in data.get("strength_hist") or []:
            g._strength_hist.append(float(v))
        for rec in data.get("history") or []:
            if not isinstance(rec, dict):
                continue
            g.history.append(Ignition(
                attractor_id=int(rec.get("attractor_id", -1)),
                strength=float(rec.get("strength", 0.0)),
                x=np.asarray(rec.get("x") or [], dtype=float),
                t_event=str(rec.get("t_event", "")),
            ))
        cur = data.get("current")
        if isinstance(cur, dict):
            g._current = Ignition(
                attractor_id=int(cur.get("attractor_id", -1)),
                strength=float(cur.get("strength", 0.0)),
                x=np.asarray(cur.get("x") or [], dtype=float),
                t_event=str(cur.get("t_event", "")),
            )
        mu = data.get("active_mu")
        if mu is not None:
            arr = np.asarray(mu, dtype=float)
            if arr.ndim == 1 and arr.size:
                g._active_mu = arr
                g._active_sigma = float(data.get("active_sigma", 0.0))
                aid = data.get("active_id")
                g._active_id = None if aid is None else int(aid)
        return g
