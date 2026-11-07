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

import numpy as np


@dataclass
class Ignition:
    """One ignition event."""
    attractor_index: int
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
        self.history: list[Ignition] = []

        self._active_mu: Optional[np.ndarray] = None
        self._active_sigma: float = 0.0
        self._active_index: Optional[int] = None

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
            self._active_index = None

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
            ref = float(np.median(self._strength_hist))
        else:
            ref = None

        self._strength_hist.append(best)

        if ref is None:
            return None

        if best <= ref or best <= 1e-12:
            return None

        a = attrs[best_i]
        self._active_index = best_i
        self._active_mu = a.mu.copy()
        self._active_sigma = float(a.sigma)

        ign = Ignition(
            attractor_index=best_i,
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
        """Median of observed peak strengths. Observation only."""
        if not self._strength_hist:
            return None
        return float(np.median(self._strength_hist))

    # Compatibility alias.
    def baseline(self) -> Optional[float]:
        return self.reference_level()

    @property
    def ignition_count(self) -> int:
        return len(self.history)
