"""
src/L4_drive/drive.py

Drive: unresolved tension between attractors.

What tension is
---------------

An edge between two attractors carries unresolved tension when both hold:

    1. its weight is above the median of all edges (high connectivity)
    2. the two centroids are separated by more than the median normalized
       separation of all edges (they are distinct regions, not one region)

Condition 1: the two attractors co-occur.
Condition 2: they are not the same region.
Together: "they belong together and are not the same thing", and this has
not been resolved into a single attractor.

Neither condition uses a fixed threshold. Both are ranks within the
system's own distribution, so the criterion moves as the system changes.

Measurement

    zw     = (w   - median_w)   / mad_w
    zs     = (sep - median_sep) / mad_sep
    amount = max(zw, 0) * max(zs, 0)

    An edge counts as tense when amount exceeds the median amount over all
    edges.

Two rejected framings
---------------------

* A drive pool with values rising toward a threshold and triggering
  behaviour. That is a fixed-rate accumulator: the rate and the threshold
  are both designer-chosen numbers.

* A reward signal. Reward requires an external standard for what counts
  as correct. No such standard exists in this system.

How tension acts
----------------

It does not accumulate and fire. It contributes a constant tilt to the
potential through `deflection()`, summed with the L2 state-field tilt.
The tilt direction is the component orthogonal to the line joining the
two centroids.

Rationale for the orthogonal direction: the tension is a relation between
two regions. A state already on that line is pushed away from it rather
than toward either end. This is the geometry of finding a third region
that accommodates both, rather than selecting one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.L3_memory.trace import TraceGraph


@dataclass
class Tension:
    """
    Unresolved tension between attractors a and b.

    amount has no upper bound and no threshold of its own; the caller
    filters by rank.
    """
    a: int
    b: int
    amount: float

    def describe(self) -> str:
        return f"{self.a}<->{self.b} tension {self.amount:.3f}"


class Drive:
    """
    Tension field.

    Reads the state space and the graph, produces a tilt vector.
    Does not trigger behaviour.
    """

    def __init__(self, dim: int = 128, gain=None):
        """
        dim  : dimensionality of the L5 state vector
        gain : magnitude of the tilt vector. None means "the structure's own
               typical gradient", supplied by set_baseline().

        On gain
        -------
        This used to be 1.0, meaning the tilt vector had unit length. That made
        the strength of the drive depend on the units of the state space: if the
        potential were measured in larger numbers, a unit-length tilt would be
        negligible; in smaller numbers, it would be overwhelming.

        The tilt enters the potential as a linear term, side by side with the
        gradient of the structure itself. Giving it the same typical magnitude
        as that gradient makes the two directly comparable: a drive of similar
        size to the structure's pull tilts the space noticeably, and one much
        smaller does not. The scale is measured from the structure, so it
        follows whatever the potential does rather than fixing a number here.

        gain sets how strongly this mechanism acts; it does not define what
        counts as tension.
        """
        self.D = int(dim)
        self.gain = float(gain) if gain is not None else None
        self._gain_measured: Optional[float] = None
        # Minimum sample sizes for the median and MAD estimates to mean
        # anything. See tensions() for why 64 rather than 8.
        self.min_edges = 64
        self.min_regions = 4
        self._last: list[Tension] = []

    def set_baseline(self, grad_norms) -> dict:
        """
        Set the tilt magnitude from the structure's own gradient scale.

        grad_norms: a sequence of ||grad V(x)|| sampled from the state space.

        The median is used rather than the mean because a single deep region
        produces a gradient far above the rest, and a mean would follow it.
        """
        arr = np.asarray([g for g in grad_norms if np.isfinite(g) and g >= 0],
                         dtype=float)
        if arr.size == 0:
            return {"ok": False, "reason": "no samples"}
        scale = float(np.median(arr))
        self._gain_measured = scale
        if self.gain is None:
            self.gain = scale
        return {"ok": True, "gain": self.gain, "median_grad": scale} 

    # ---------- tension ----------

    def tensions(self, state_space, graph: TraceGraph) -> list[Tension]:
        """
        Unresolved tensions at the current structure.

        The threshold is set by what the measurement needs, not by a wish to
        avoid small numbers.

        Both the weight scale and the separation scale are estimated with the
        median and the median absolute deviation, and the tensions are their
        z-scores. The MAD has a relative standard error of roughly 0.8/sqrt(n),
        so reaching ten percent accuracy needs on the order of 64 samples.
        Below that the z-scores are dominated by estimation error rather than
        by the structure.

        Measured on a synthetic structure while varying the edge count:

            edges  tensions  mean amount
               14         2        1.31
               29         3       15.61
              213        30        7.90
              492        56        2.86

        At 29 edges the mean amount is five times what it is at 213, which is
        the MAD being too small to normalise by, not the structure being more
        tense. The old floor of 8 edges let that through.

        Regions: four is the point at which a pair can be separated from the
        rest of the set at all; below that the separation z-scores have no
        population to be relative to.
        """
        attrs = state_space.attractors()
        n = len(attrs)
        # Graph nodes are uids; the numbers live in parallel arrays indexed by
        # row. `uid_row()` is cached per structural version, so this lookup
        # costs nothing per step, and an edge whose endpoint is gone is dropped
        # here rather than silently reading a different region's centroid.
        row = state_space.uid_row()
        edges = [e for e in graph.edges() if e.a in row and e.b in row]

        if len(edges) < self.min_edges or n < self.min_regions:
            self._last = []
            return []

        # Vectorised: pull centroids, scales and weights once, then compute
        # all separations in one array operation. A Python loop over edges was
        # the single largest cost in a step after potential/gradient were
        # vectorised.
        idx_a = np.fromiter((row[e.a] for e in edges), dtype=np.int64, count=len(edges))
        idx_b = np.fromiter((row[e.b] for e in edges), dtype=np.int64, count=len(edges))
        # The two ends are carried separately in uid space: a Tension names two
        # regions, and a row number would stop meaning the same region after the
        # next structural change. (Rows are for the arrays only.)
        uid_a = np.fromiter((e.a for e in edges), dtype=np.int64, count=len(edges))
        uid_b = np.fromiter((e.b for e in edges), dtype=np.int64, count=len(edges))
        ws = np.fromiter((e.w for e in edges), dtype=float, count=len(edges))

        mus = np.stack([a.mu for a in attrs])
        sig = np.maximum(
            np.array([a.sigma for a in attrs], dtype=float),
            np.finfo(float).eps,
        )

        d = mus[idx_a] - mus[idx_b]
        sep = np.sqrt(np.einsum("ij,ij->i", d, d)) / np.maximum(sig[idx_a], sig[idx_b])

        w_med = float(np.median(ws))
        w_mad = float(np.median(np.abs(ws - w_med)))
        if w_mad < 1e-12:
            w_mad = max(abs(w_med), float(np.sqrt(np.finfo(float).eps)))

        s_med = float(np.median(sep))
        s_mad = float(np.median(np.abs(sep - s_med)))
        if s_mad < 1e-12:
            s_mad = max(abs(s_med), float(np.sqrt(np.finfo(float).eps)))

        zw = (ws - w_med) / w_mad
        zs = (sep - s_med) / s_mad

        # Geometric mean rather than a product.
        #
        # Tension is defined as high on both axes at once: a connection that is
        # both strong and long. The product of the two z-scores does express
        # that, but it grows with the square of whichever is larger, so one
        # outlier counts for far more than twice what a moderate pair does. Two
        # z-scores of 10 give 100; two of 5 give 25, four times less for half
        # the extremity on each axis. The geometric mean keeps the conjunction
        # and makes the amount grow in proportion to the extremes rather than to
        # their product.
        #
        # The factor of two inside the mean is the normalisation that makes a
        # pair at the median on both axes score 1.0: (1*1)^(1/2) = 1.
        a_w = np.maximum(zw, 0.0)
        a_s = np.maximum(zs, 0.0)
        amount = np.sqrt(a_w * a_s)

        positive = amount > 0.0
        if not positive.any():
            self._last = []
            return []

        ref = float(np.median(amount[positive]))
        keep_mask = positive & (amount > ref)
        if not keep_mask.any():
            self._last = []
            return []

        order = np.argsort(-amount[keep_mask])
        ka = uid_a[keep_mask][order]
        kb = uid_b[keep_mask][order]
        kv = amount[keep_mask][order]

        keep = [Tension(a=int(a), b=int(b), amount=float(v))
                for a, b, v in zip(ka, kb, kv)]
        self._last = keep
        return keep

    # ---------- tilt vector ----------

    def deflection(self, state_space, graph: TraceGraph) -> Optional[np.ndarray]:
        """
        Sum of unit vectors orthogonal to each tension pair, weighted by amount,
        normalized to unit length, scaled by gain.

        Normalization prevents the tilt magnitude from growing with the number
        of attractors.
        """
        ts = self.tensions(state_space, graph)
        if not ts:
            return None

        attrs = state_space.attractors()
        row = state_space.uid_row()
        acc = np.zeros(self.D, dtype=float)
        total = 0.0

        for t in ts:
            # A tension names two regions by uid; an edge that survived a
            # structural change points at a region that still exists, so a
            # miss here means the graph is stale and the pair is skipped
            # rather than read as some other region's centroid.
            ia = row.get(int(t.a))
            ib = row.get(int(t.b))
            if ia is None or ib is None:
                continue
            d = attrs[ia].mu - attrs[ib].mu
            nd = float(np.linalg.norm(d))
            if nd <= 0:
                continue
            d = d / nd

            perp = self._perpendicular(d)
            if perp is None:
                continue
            acc += t.amount * perp
            total += abs(t.amount)

        if total <= 0:
            return None

        m = float(np.linalg.norm(acc))
        if m <= 0:
            return None

        # Without a measured baseline the tilt has no scale, and a unit-length
        # tilt would be an assumption about the units of the state space.
        # Returning nothing is honest: the caller has not supplied the scale.
        if self.gain is None:
            return None
        return acc / m * self.gain

    def _perpendicular(self, v: np.ndarray) -> Optional[np.ndarray]:
        """Unit vector orthogonal to v, via Gram-Schmidt on the least-aligned axis."""
        i = int(np.argmin(np.abs(v)))
        e = np.zeros_like(v)
        e[i] = 1.0
        p = e - float(e @ v) * v
        m = float(np.linalg.norm(p))
        if m <= 0:
            return None
        return p / m

    # ---------- observation ----------

    @property
    def last(self) -> list[Tension]:
        return list(self._last)

    def summary(self) -> dict:
        ts = self._last
        if not ts:
            return {"n": 0, "max": 0.0, "sum": 0.0}
        return {
            "n": len(ts),
            "max": round(ts[0].amount, 4),
            "sum": round(sum(t.amount for t in ts), 4),
        }
