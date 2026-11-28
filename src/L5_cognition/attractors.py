"""
src/L5_cognition/attractors.py

LEGACY. Not used by the runtime pipeline.

This is an earlier attractor implementation based on k-dimensional
subspaces and Mahalanobis distance. It has been superseded by
terrain.py (StateSpace / Attractor), which uses full-dimensional
Gaussian components and a measured characteristic scale.

Kept only because scripts/_diag_att.py imports it.

Differences from the current implementation
-------------------------------------------

attractors.py (legacy)         terrain.py (current)
---------------------------    ------------------------------------
subspace basis per attractor   full-dimensional Gaussian
Mahalanobis distance           normalized Euclidean distance
depth set by L3 update         depth accumulated from observations
k = 32 by default              D = 128, no subspace

Terms used here follow the current terminology: an attractor is a
recurring region of state space, not a basin of attraction.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Attractor7:
    """
    Legacy attractor defined in a k-dimensional subspace.

    Attractor IDs are integers; no semantic label is ever assigned.
    """
    attractor_id: int
    basis: np.ndarray          # (k, D) orthonormal basis, k <= D
    center: np.ndarray         # (k,) coordinates in the subspace
    cov: np.ndarray            # (k, k) covariance within the subspace
    origin: np.ndarray         # (D,) subspace reference point (sample mean)
    depth: float = 0.0         # potential depth, accumulated from observations
    visits: int = 0
    born_at: Optional[str] = None

    @property
    def k(self) -> int:
        return self.center.shape[0]

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project a full-space state into this attractor's subspace.

        The origin must be subtracted: basis and center are defined
        relative to it.
        """
        return self.basis @ (x - self.origin)

    def distance(self, x: np.ndarray) -> float:
        """Mahalanobis distance; falls back to Euclidean if the covariance is singular."""
        p = self.project(x)
        d = p - self.center
        try:
            solved = np.linalg.solve(self.cov + np.eye(self.k) * 1e-6, d)
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(d))
        return float(np.sqrt(max(d @ solved, 0.0)))

    def potential(self, x: np.ndarray) -> float:
        """Contribution of this attractor to the potential at x (negative)."""
        if self.depth <= 1e-9:
            return 0.0
        d = self.distance(x)
        return -self.depth * float(np.exp(-0.5 * d * d))


