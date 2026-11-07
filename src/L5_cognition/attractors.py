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


class AttractorSet:
    """
    LEGACY container of subspace attractors.

    Responsibilities: ID allocation, growth from state samples, decay,
    potential queries.
    """

    def __init__(self, dim: int = 128, subspace_dim: int = 32):
        self.D = int(dim)
        self.k = int(subspace_dim)
        self._items: dict[int, Attractor7] = {}
        self._next_id = 0

    # ---------- query ----------

    def get(self, attractor_id: int) -> Optional[Attractor7]:
        return self._items.get(attractor_id)

    def all(self) -> List[Attractor7]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def potential(self, x: np.ndarray) -> float:
        return float(sum(a.potential(x) for a in self._items.values()))

    def nearest(self, x: np.ndarray) -> Optional[tuple[int, float]]:
        """(nearest attractor id, distance); None if there are no attractors."""
        best = None
        for a in self._items.values():
            d = a.distance(x)
            if best is None or d < best[1]:
                best = (a.attractor_id, d)
        return best

    # ---------- growth ----------

    def grow(self, samples: np.ndarray, born_at: str | None = None) -> Attractor7:
        """
        Create an attractor from state samples.

        samples: (n, D). Its mean is the origin, its principal directions
        define the subspace.
        """
        X = np.asarray(samples, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.D:
            raise ValueError(f"samples must be (n, {self.D}), got {X.shape}")

        mu = X.mean(axis=0)
        Xc = X - mu
        cov_full = (Xc.T @ Xc) / max(len(X) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov_full)
        order = np.argsort(eigvals)[::-1][: self.k]
        basis = eigvecs[:, order].T

        proj = Xc @ basis.T
        center = proj.mean(axis=0)
        cov_k = (proj - center).T @ (proj - center) / max(len(X) - 1, 1)
        cov_k += np.eye(self.k) * 1e-6

        att = Attractor7(
            attractor_id=self._next_id,
            basis=basis,
            center=center,
            cov=cov_k,
            origin=mu,
            depth=0.0,
            born_at=born_at,
        )
        self._items[self._next_id] = att
        self._next_id += 1
        return att

    # ---------- modification ----------

    def deepen(self, attractor_id: int, amount: float, cap: float = 1.0) -> None:
        a = self._items.get(attractor_id)
        if a is not None:
            a.depth = float(min(a.depth + amount, cap))

    def decay(self, factor: float) -> None:
        """Apply decay to all attractors; remove those below 1e-4."""
        for aid in list(self._items.keys()):
            a = self._items[aid]
            a.depth *= factor
            if a.depth < 1e-4:
                del self._items[aid]

    def remove(self, attractor_id: int) -> None:
        self._items.pop(attractor_id, None)
