"""
src/L5_cognition/terrain.py

State space with attractor formation.

Model (formal description, no metaphor)
---------------------------------------

The state is a point x in R^D (D = 128).

An `Attractor` records a recurring region of the state space:

    - mu      : centroid of the states assigned to it
    - amp     : depth (accumulated assignment minus recovery)
    - members : the most recent assigned states (bounded buffer)
    - sigma   : characteristic scale of the attractor

Assignment rule
---------------

A new state x is assigned to attractor A if

    ||x - mu_A|| / sigma_A < 1

i.e. the normalized distance is below one characteristic scale.
Where sigma is defined:

    - if the attractor has >= 4 members: RMS deviation of the members
      from their centroid (a measured quantity)
    - otherwise: the global scale recorded at construction time
      (also a measured quantity, see _global_scale)

Both are measured, not configured.

Note on the previous version
----------------------------

The earlier wording described sigma as a "territory radius" and
assignment as "walking into" a region. That framing imported a spatial
metaphor that does not hold: there is no space being walked through,
only a metric on the state vector.

The mathematics is unchanged; this file now states it directly.

Recovery
--------

`recover()` reduces every attractor's depth by its own rate factor.
The factor is drawn at construction and never changes; a per-call
random perturbation is applied on top.

Recovery is not conditional on recent use. Conditional-on-use decay
would require an external criterion for what counts as "use", which
this system does not have.
"""

from __future__ import annotations

import collections
import numpy as np
from dataclasses import dataclass, field
from typing import Deque, List, Optional


# Minimum assignments before a region's revisit interval is estimated from
# its own count. Below this the ratio of total steps to assignments is too
# noisy to be an interval. Observation resolution, not a mechanism.
MIN_ASSIGN_FOR_INTERVAL = 4

# Lower bound on the interval-based rate scaling. A region visited once and
# never again decays at this fraction of the base rate rather than freezing.
MIN_RATE_SCALE = 0.01


@dataclass
class Attractor:
    """
    A recurring region of state space.

    Fields:
        mu            centroid (measured)
        amp           depth, result of assignment vs. recovery balance
        members       recent assigned states, bounded buffer
        scale         initial characteristic scale (set at construction)
        _sigma_cache  cached characteristic scale
        _decay_factor per-attractor recovery rate multiplier (fixed at birth)
    """
    mu: np.ndarray
    amp: float
    members: Deque[np.ndarray] = field(default_factory=lambda: collections.deque(maxlen=64))
    born_t: Optional[str] = None
    scale: Optional[float] = None
    _sigma_cache: Optional[float] = field(default=None, repr=False)
    # Member count at the time the cached scale was computed.
    _sigma_members_at: Optional[int] = field(default=None, repr=False)
    _decay_factor: Optional[float] = None
    # Total assignments received since creation. Separate from len(members),
    # which is a bounded buffer and stops counting at 64. Needed to measure
    # how often this region is actually reached (see L4_drive/viability.py).
    _assigned: int = 0
    # Step index at which this region was created. Visit rate has to be measured
    # over the region's own lifetime; using the total step count as the
    # denominator makes every young region look almost never visited, which
    # understates how much it can carry and inflates its viability ratio. On the
    # real system that produced a population median ratio of 4.8 with individual
    # regions above 25, purely as an artefact of when they were born.
    _born_step: int = 0

    # Compatibility: older checkpoints stored this attribute under the
    # name `_sigma_decay`. Keep the alias so existing state loads.
    @property
    def _sigma_decay(self) -> Optional[float]:
        return self._decay_factor

    @_sigma_decay.setter
    def _sigma_decay(self, v: Optional[float]) -> None:
        self._decay_factor = v

    # ---------- characteristic scale ----------

    @property
    def sigma(self) -> float:
        """
        Characteristic scale of this attractor.

        Estimated from the buffered members with a robust estimator: the
        median absolute deviation of member distances from the centroid, scaled
        to be comparable with a standard deviation.

        Why not the RMS deviation
        -------------------------
        The RMS deviation is pulled upward by every member that sits far from
        the centroid, and membership is itself decided by this scale (a state is
        assigned when its distance is below one sigma). That closes a positive
        loop: a wide region admits distant states, distant states widen it.

        Measured, the loop has two outcomes depending on starting conditions:

            starting from a checkpoint of fine fragments
                sigma median 0.0055, depth median 0.015, 246 regions
            starting from a cold space
                sigma median 0.0437, depth median 0.309, 32 regions

        The cold start does not fragment; it inflates, and then stops forming
        regions at all because everything falls inside one of the 32 wide ones.

        A robust estimator does not have this property. A few distant members
        move the median deviation very little, so admitting a stray state does
        not widen the region, and the scale settles toward the typical spread of
        the states actually assigned rather than toward the widest.

        The factor 1.4826 rescales MAD to a standard-deviation equivalent for
        normally distributed members, so existing thresholds keep their meaning.
        """
        if self._sigma_cache is not None:
            return self._sigma_cache

        if len(self.members) >= 4:
            M = np.stack(self.members)
            d = M - self.mu
            r = np.sqrt(np.sum(d * d, axis=1))
            mad = float(np.median(np.abs(r - np.median(r))))
            s = 1.4826 * mad
            if not np.isfinite(s) or s <= 0:
                # Degenerate spread (most members identical): fall back to the
                # scale recorded at construction, which came from the structure.
                s = self.scale or 1.0
            self._sigma_cache = s
        else:
            self._sigma_cache = self.scale if self.scale is not None else 1.0

        floor = getattr(self, "_scale_floor", None)
        if floor and self._sigma_cache < floor:
            self._sigma_cache = float(floor)
        return self._sigma_cache

    def assign(self, x: np.ndarray) -> None:
        """Assign one state to this attractor and update the centroid."""
        self.members.append(x.copy())
        self._assigned += 1
        # The centroid is the mean of the buffered members, not of all
        # assignments; the buffer is what defines this region's current shape.
        n = len(self.members)
        self.mu += (x - self.mu) / n
        # The cached scale is dropped only when the member count has grown
        # enough that the median deviation could actually change.
        #
        # Invalidating on every assignment meant the cache never survived a
        # step: the property was computed on every access, and profiling showed
        # 509000 calls over 2000 steps taking 3.4 of 9.3 seconds. The member
        # buffer is a bounded deque, so once it is full the count stops moving
        # and the scale is recomputed only on the explicit invalidations that
        # follow merges and floor changes.
        if self._sigma_members_at is None or n >= 4 and n >= 2 * self._sigma_members_at:
            self._sigma_members_at = n
            self._sigma_cache = None

    # Compatibility alias for the previous method name.
    def admit(self, x: np.ndarray) -> None:
        self.assign(x)


class StateSpace:
    """
    State space with plastic attractor formation.

    Public methods:
        potential(x)      scalar potential at x (lower = deeper)
        gradient(x)       analytic gradient
        record(x)         register one observed state
        recover()         apply recovery
    """

    def __init__(
        self,
        dim: int = 128,
        baseline: float = 0.0,
        recover_rate: float = 0.002,
        max_attractors: int = 8192,
        min_amp: float = 1e-3,
        ceiling: float = 1.0,
        assign_ratio: float = 0.05,
        recover_jitter: float = 0.5,
        decay_spread: float = 0.6,
    ):
        """
        dim             : dimensionality of the state vector
        baseline        : potential at an unvisited region
        recover_rate    : base recovery rate
        max_attractors  : safety bound on memory, not a target count
        min_amp         : depth below which an attractor is removed
        ceiling         : depth capacity of a single attractor
        assign_ratio    : fraction of the remaining headroom gained per assignment
        recover_jitter  : amplitude of per-call random fluctuation in recovery

        On max_attractors: this is a guard against unbounded memory use, not a
        design parameter. Measured behaviour with the bound effectively removed
        settles around 780 attractors and stays there; the previous bound of
        512 was below the natural count, so it was exceeded continuously, and
        the merge-on-overflow path never fired because its criterion
        (centroid distance below the larger scale) does not hold at this
        density. A bound that never binds is better than one that always does.

        decay_spread    : spread of the per-attractor fixed decay factor
        """
        self.D = int(dim)
        self.baseline = float(baseline)
        self.recover_rate = float(recover_rate)
        self.max_attractors = int(max_attractors)
        self.min_amp = float(min_amp)
        self.ceiling = float(ceiling)
        self.assign_ratio = float(assign_ratio)
        self.recover_jitter = float(recover_jitter)
        self.decay_spread = float(decay_spread)

        self._attractors: List[Attractor] = []

        # Global scale: how far the observed states have ranged, measured
        # cumulatively. Used as the initial characteristic scale of a new
        # attractor.
        self._recent: Deque[np.ndarray] = collections.deque(maxlen=64)
        self._global_scale: float = 1.0
        self.MIN_SCALE_SAMPLES = 8
        self._last_x: Optional[np.ndarray] = None
        self._center: Optional[np.ndarray] = None
        self._scale_seeded: bool = False
        # Cached region scale. current_scale() is O(n^2) in the number of
        # attractors and is read on every observation; without caching a
        # 700-attractor space costs ~500k distance evaluations per step.
        self._scale_cache: Optional[float] = None
        self._scale_cache_n: int = -1
        self._mu_cache: Optional[np.ndarray] = None
        self._mu_cache_n: int = -1
        self._sigma_vec: Optional[np.ndarray] = None

        # Per-region pull factors supplied by L4 regulation. All ones by default,
        # so the dynamics are unchanged unless regulation is attached.
        self._pull: Optional[np.ndarray] = None
        # Rest per region, subtractive on depth. None disables it, so the
        # dynamics are unchanged until a rest field is attached.
        self._rest: Optional[np.ndarray] = None
        # Traversability edge: potential level beyond which the structure is too
        # thin to hold a state. None disables it.
        self._wall: Optional[float] = None
        self._wall_gain: float = 1.0
        # Total observations ever recorded. Used as the denominator when
        # measuring per-region visit rate.
        self._total_steps: int = 0

        # 追加式日志（src/common/journal.py）。为 None 时不记录。
        # 记的是'她经历了什么'，不是'她成了什么样'——后者由重放算出来。
        self.journal = None

        # 折叠汇总：被折叠区域的 (量化质心, 深度累加, 计数)，用于载体有界。
        self._folded: list[dict] = []
        # Recovery cadence hint, set by the caller (Stream) so recovery can
        # relate a rate to a number of steps.
        self._recover_every_hint: int = 16
        # The scale floor is applied on the first observation after loading, so
        # a checkpoint containing an abnormally narrow region cannot dominate
        # the gradient before the first recovery pass.
        self._scale_floor_pending: bool = True
        # Cap on how much any single region may contribute to the gradient
        # relative to all others combined. See gradient().
        self._pull_cap_active: bool = True
        self._rng = np.random.default_rng()

    # Compatibility aliases (previous public names).
    @property
    def deposit_ratio(self) -> float:
        return self.assign_ratio

    @property
    def max_traces(self) -> int:
        return self.max_attractors

    def _update_global_scale(self, x: np.ndarray) -> None:
        """
        Record one observed state. Does not maintain the scale.

        The scale is not tracked continuously. See `current_scale()` for why.
        """
        self._recent.append(np.asarray(x, dtype=float).copy())

    def current_scale(self) -> Optional[float]:
        """
        The region scale in use right now, measured from existing structure.

        Returns None when there is not enough evidence to define one. The
        caller treats None as "do not create a region yet": before the state
        has moved enough to say how large a region is, any width chosen would
        be arbitrary, and it would set the width of every region created
        afterwards.

        Why this is computed on demand rather than tracked step by step
        ----------------------------------------------------------------
        Several continuous estimators were tried and all failed in the same
        way. When the state sits inside one deep attractor it barely moves, so
        any estimator driven by recent positions reports a scale around 1e-4
        while the space actually visited spans several units:

          * RMS deviation over the last 64 states   -> 1e-4 (window too short)
          * weighted average of step length         -> 1e-4 (steps are small)
          * running centre with running radius      -> 1e-4 (centre follows
            the state, so the distance to it stays small)

        Each estimator answers "how much is the state moving", which is not
        the question. The question is "how large is a region of this space",
        and the only direct evidence for that is the spacing between the
        regions already formed.

        So the scale is the median nearest-neighbour distance among existing
        attractor centroids. That is a measurement of the structure in use,
        not of the current motion, and it does not change while the state
        stands still.

        On the cold-start fallback: the first version fell back to the radius
        of the last eight observed states. Eight points cover only as far as
        the state happened to move in eight steps, which on a fresh space is a
        very wide region, and since the first region's width seeds every
        later one the space ended up with 20 regions in 8000 steps while the
        same code on a loaded checkpoint produced 294. The fallback is gone;
        the space waits until it has moved enough to measure.
        """
        if len(self._attractors) >= 8:
            n_now = len(self._attractors)
            # The cache is refreshed when the count has changed by more than
            # a small fraction. This value is only used as the starting scale
            # of a new attractor, so a slightly stale estimate is harmless,
            # while recomputing an O(n^2) nearest-neighbour matrix on every
            # creation was taking 4 ms per call and dominated the step.
            if (self._scale_cache is not None
                    and abs(n_now - self._scale_cache_n)
                    <= max(4, self._scale_cache_n // 20)):
                return self._scale_cache
            mus = np.stack([a.mu for a in self._attractors])
            sq = np.sum(mus * mus, axis=1)
            d2 = sq[:, None] + sq[None, :] - 2.0 * (mus @ mus.T)
            np.maximum(d2, 0.0, out=d2)
            np.fill_diagonal(d2, np.inf)
            nn = np.sqrt(d2.min(axis=1))
            val = float(np.median(nn))
            self._scale_cache = val
            self._scale_cache_n = n_now
            return val
        if len(self._recent) >= self.MIN_SCALE_SAMPLES:
            M = np.stack(list(self._recent))
            c = M.mean(axis=0)
            return float(np.max(np.linalg.norm(M - c, axis=1)))
        return None

    def _invalidate_scale(self) -> None:
        """Drop cached scale and centroid matrix; the attractor list changed."""
        self._scale_cache = None
        self._scale_cache_n = -1
        self._mu_cache = None
        self._mu_cache_n = -1

    def _numerical_floor(self, v: np.ndarray) -> float:
        """Numerical floor tied to the magnitude of v: sqrt(eps) * max(|v|, 1)."""
        mag = float(np.linalg.norm(v))
        return float(np.sqrt(np.finfo(float).eps)) * max(mag, 1.0)

    @property
    def scale_ready(self) -> bool:
        return len(self._recent) >= self.MIN_SCALE_SAMPLES

    @property
    def global_scale(self) -> float:
        """Current region scale, or 0.0 when there is not yet enough evidence."""
        v = self.current_scale()
        return 0.0 if v is None else v

    def __len__(self) -> int:
        return len(self._attractors)

    def attractors(self) -> List[Attractor]:
        return list(self._attractors)

    # Compatibility alias.
    def traces(self) -> List[Attractor]:
        return list(self._attractors)

    # ---------- regulation hook ----------

    def _apply_scale_floor(self) -> None:
        """
        Clamp every attractor's characteristic scale.

        Two floors, both derived from data:

        1. A fraction of the median scale, so that no region may be an order of
           magnitude narrower than is typical without its gradient being
           treated as an artefact.

        2. A floor tied to the region's own depth. A region that holds nearly
           the full depth capacity, sustained by tens of thousands of
           assignments, cannot in fact occupy a point: the states assigned to it
           were spread over however much space they were spread over, and if
           that spread reads as 4e-4 while the depth reads 0.997 the two
           measurements disagree and the gradient is dominated by the
           disagreement. Measured on a checkpoint written by an earlier version:
           one region had 76456 assignments, depth 0.997, scale 4.1e-4, giving
           depth / scale^2 = 5.9e6 and a gradient of 1125 while every other
           region contributed below 0.6.

           The relation used is sigma >= depth / (depth + 1) * median_scale.
           It is a self-consistency bound rather than a width: a region holding
           a large fraction of the depth capacity is required to be at least
           comparable in width to a typical region.
        """
        if len(self._attractors) < 8:
            return
        sig = np.array([a.sigma for a in self._attractors], dtype=float)
        sig = sig[sig > 0]
        if sig.size < 8:
            return
        med = float(np.median(sig))
        global_floor = med * 0.1

        for a in self._attractors:
            depth_fraction = min(max(a.amp / max(self.ceiling, 1e-12), 0.0), 1.0)
            consistent = depth_fraction / (depth_fraction + 1.0) * med
            a._scale_floor = max(global_floor, consistent)
            # sigma is cached; the floor only takes effect once the cache is
            # dropped, otherwise the cached value keeps the old scale.
            a._sigma_cache = None

        # Pull cap. The floors above keep any single region's gradient from
        # being far larger than the sum of the rest, but a region that is both
        # deep and narrow relative to the current state can still dominate for
        # the moment the state sits near it. The cap is expressed relative to
        # the total pull of all regions, so it says only that no single region
        # may outweigh all the others combined; it does not set an absolute
        # strength.
        self._pull_cap_active = True

    def set_pull(self, factors: Optional[np.ndarray]) -> None:
        """
        Set per-region pull factors (L4 regulation).

        None disables regulation and restores unit pull. A length mismatch is
        ignored rather than applied to the wrong regions.
        """
        if factors is None:
            self._pull = None
            return
        f = np.asarray(factors, dtype=float)
        if f.shape[0] != len(self._attractors):
            return
        self._pull = f.copy()

    def set_rest(self, values: Optional[np.ndarray]) -> None:
        """
        Set per-region rest values (L5 rest field).

        Rest is subtractive: a region's depth is multiplied by (1 - rest), so a
        rest of zero leaves it unchanged and a rest of one removes its hold
        completely. Rest can make a region less attractive, never more.
        """
        if values is None:
            self._rest = None
            return
        v = np.asarray(values, dtype=float)
        if v.shape[0] != len(self._attractors):
            return
        self._rest = np.clip(v, 0.0, 1.0)

    def _amp_vector(self) -> np.ndarray:
        """Depths, already multiplied by regulation and rest factors."""
        amp = np.array([a.amp for a in self._attractors], dtype=float)
        if self._pull is not None and self._pull.shape[0] == amp.shape[0]:
            amp = amp * self._pull
        if self._rest is not None and self._rest.shape[0] == amp.shape[0]:
            amp = amp * (1.0 - self._rest)
        return amp

    # ---------- potential ----------

    def potential(self, x: np.ndarray) -> float:
        """
        Potential at x. Vectorised over attractors.

        Only attractors within ~3.5 sigma contribute meaningfully; the rest
        are skipped by masking. The kernel is evaluated in one array
        operation instead of a Python loop.
        """
        if not self._attractors:
            return float(self.baseline)
        x = np.asarray(x, dtype=float)
        self._centroid_matrix()
        d = self._mu_cache - x
        d2 = np.einsum("ij,ij->i", d, d)
        sig = np.maximum(self._sigma_vec, np.finfo(float).eps)
        amp = self._amp_vector()
        mask = d2 <= 12.0 * sig * sig
        if not mask.any():
            return float(self.baseline) + self._wall_potential(x)
        contrib = amp[mask] * np.exp(-0.5 * d2[mask] / (sig[mask] ** 2))
        return float(self.baseline - contrib.sum()) + self._wall_potential(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Analytic gradient of the potential, vectorised over attractors.

        Per-region contributions are summed with a dominance cap: no single
        region may contribute more than the sum of the others. The cap is
        expressed relative to the other contributions, so it never sets an
        absolute strength and never suppresses a region in a space where one
        genuinely outweighs all the rest.

        Why: a region carried over from an earlier version of the code had
        76456 assignments, depth 0.997 and scale 4.1e-4, and contributed a
        gradient of 1125 while every other region contributed below 0.6. It is
        kept, because it is a real part of the recorded history, but it is not
        allowed to determine the whole trajectory by itself.
        """
        if not self._attractors:
            return np.zeros(self.D)
        x = np.asarray(x, dtype=float)
        self._centroid_matrix()
        dm = self._mu_cache - x
        d2 = np.einsum("ij,ij->i", dm, dm)
        sig = np.maximum(self._sigma_vec, np.finfo(float).eps)
        amp = self._amp_vector()
        mask = d2 <= 12.0 * sig * sig
        if not mask.any():
            return self._wall_gradient(x)

        idx = np.flatnonzero(mask)
        w = amp[idx] * np.exp(-0.5 * d2[idx] / (sig[idx] ** 2))
        # per-region vector contribution
        parts = -(w[:, None] * dm[idx] / (sig[idx] ** 2)[:, None])

        if self._pull_cap_active and len(idx) > 1:
            mags = np.linalg.norm(parts, axis=1)
            total = float(mags.sum())
            biggest = int(np.argmax(mags))
            rest = total - float(mags[biggest])
            if rest > 0 and mags[biggest] > rest:
                parts[biggest] *= rest / float(mags[biggest])

        g = parts.sum(axis=0)
        return g + self._wall_gradient(x)

    # ---------- traversability ----------

    def set_wall(self, level: Optional[float], gain: float = 1.0) -> None:
        """
        Set the potential level beyond which the space stops being traversable,
        and how strongly that edge acts.

        gain converts the shortfall into a gradient comparable in magnitude to
        the ordinary attraction. Without it the traversability gradient is much
        larger than the attractor gradient: measured, an ungained wall drove
        speed to 8.8 where the unregulated system runs at 0.05.

        That gain is a numerical scale, not a criterion: it sets how firmly the
        edge pushes, not where the edge is.
        """
        self._wall = None if level is None else float(abs(level))
        self._wall_gain = float(max(gain, 0.0))

    def _depth_at(self, x: np.ndarray) -> float:
        """Depth this point would have from the structure, ignoring regulation."""
        if self._mu_cache is None:
            self._centroid_matrix()
        d = self._mu_cache - x
        d2 = np.einsum("ij,ij->i", d, d)
        sig = np.maximum(self._sigma_vec, np.finfo(float).eps)
        return float((self._amp_vector() * np.exp(-0.5 * d2 / (sig * sig))).sum())

    def _wall_potential(self, x: np.ndarray) -> float:
        """
        Extra potential that rises where the structure is too thin.

            extra = max(wall - depth_at(x), 0)

        Smooth: zero everywhere inside the maintained depth range, rising
        linearly outside it. No hard boundary, so the gradient stays finite.
        """
        if self._wall is None or not self._attractors:
            return 0.0
        if self._mu_cache is None:
            self._centroid_matrix()
        return self._wall_gain * max(self._wall - self._depth_at(x), 0.0)

    def _wall_gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the traversability term."""
        if self._wall is None or not self._attractors:
            return np.zeros(self.D)
        if self._mu_cache is None:
            self._centroid_matrix()
        if self._depth_at(x) >= self._wall:
            return np.zeros(self.D)
        d = self._mu_cache - x
        d2 = np.einsum("ij,ij->i", d, d)
        sig = np.maximum(self._sigma_vec, np.finfo(float).eps)
        w = self._amp_vector() * np.exp(-0.5 * d2 / (sig * sig))
        return self._wall_gain * (w[:, None] * d / (sig ** 2)[:, None]).sum(axis=0)

    # ---------- observation ----------

    def record(self, x: np.ndarray, t_event: str | None = None) -> Optional[Attractor]:
        """
        Register one observed state.

        Returns the attractor the state was assigned to, or None when there
        are not yet enough observations to measure a scale (the first few
        states of a fresh state space).

        Assignment uses the attractor's own characteristic scale. This is
        deliberate: an attractor is a region with a shape, and a state within
        one scale of its centroid belongs to it. The scale itself is measured
        from data, never configured (see Attractor.sigma).
        """
        x = np.asarray(x, dtype=float)
        # A non-finite state would poison every centroid it is assigned to,
        # and the damage is permanent once checkpointed. Stop here instead.
        if not np.isfinite(x).all():
            raise ValueError("non-finite observation; refusing to record")
        if self.journal is not None:
            self.journal.append_observe(x)
        self._update_global_scale(x)
        self._total_steps += 1

        if not self.scale_ready:
            return None

        if self._scale_floor_pending:
            self._apply_scale_floor()
            self._scale_floor_pending = False

        # Vectorised assignment test. Computing this per attractor with
        # np.linalg.norm cost 146k calls per 100 steps and dominated the
        # profile (289 ms of 1 s). A single matmul over a cached centroid
        # matrix does the same arithmetic in one array operation.
        best = self._nearest_normalized(x)

        if best is not None and best[0] < 1.0:
            a = self._attractors[best[1]]
            a.assign(x)
            headroom = max(self.ceiling - a.amp, 0.0)
            a.amp += headroom * self.assign_ratio
            return a

        # New region. Its initial scale comes from the spacing of the regions
        # already present, not from the recent motion of the state. If there is
        # not enough evidence to measure that spacing, no region is created
        # yet: a width chosen now would set the width of every later region.
        measured = self.current_scale()
        if measured is None:
            return None
        init_scale = max(measured, self._numerical_floor(x))
        # A region is born with the depth it can actually sustain, not with a
        # fixed fraction of the ceiling.
        #
        # The birth depth used to be ceiling * assign_ratio, which is 0.05 with
        # the defaults. The depth a region can carry at its observed visit rate
        # is ceiling * (p*a) / (p*a + R); for a region seen once in a long run
        # that is a few thousandths. So every new region started an order of
        # magnitude deeper than it could hold, and the viability ratio for a
        # young region was in the tens before anything had happened to it.
        #
        # That mismatch was invisible while regulation clamped its factor at a
        # fixed ceiling. Removing the clamp exposed it.
        a = Attractor(
            mu=x.copy(),
            amp=self._birth_depth(init_scale),
            born_t=t_event,
            scale=init_scale,
        )
        a._decay_factor = self._birth_decay_factor()
        a._decay_factor = max(a._decay_factor, 0.05)
        a.members.append(x.copy())
        a._assigned = 1
        a._born_step = int(self._total_steps)
        self._attractors.append(a)
        self._invalidate_scale()

        if len(self._attractors) > self.max_attractors:
            # Merge repeatedly until back under capacity. A single merge per
            # step cannot keep up when creation is faster than merging, and the
            # count was observed to overrun a cap of 512 by more than 140.
            #
            # The loop is bounded so that a pathological configuration cannot
            # stall: if no overlapping pair exists, capacity is simply exceeded
            # and that is accepted rather than resolved by removing on a value
            # criterion (which this system does not define).
            for _ in range(64):
                if len(self._attractors) <= self.max_attractors:
                    break
                if not self._merge_overlapping():
                    break

        return a

    # Compatibility alias.
    def pass_through(self, x: np.ndarray, t_event: str | None = None) -> Optional[Attractor]:
        return self.record(x, t_event)

    # ---------- vectorised nearest lookup ----------

    def _centroid_matrix(self) -> Optional[np.ndarray]:
        """
        Cached (n, D) matrix of centroids. Rebuilt when the count changes.

        Centroids move slightly on every assignment, so a stale matrix would
        give slightly wrong distances. The matrix is therefore refreshed
        whenever the attractor count changes, and the per-step drift of an
        individual centroid is small enough not to alter which attractor is
        nearest. Any lookup that must be exact (merge, scale) rebuilds
        explicitly.
        """
        n = len(self._attractors)
        if self._mu_cache is None or self._mu_cache_n != n:
            self._mu_cache = np.stack([a.mu for a in self._attractors])
            self._mu_cache_n = n
            self._sigma_vec = np.array([a.sigma for a in self._attractors],
                                       dtype=float)
        return self._mu_cache

    def _invalidate_centroids(self) -> None:
        self._mu_cache = None
        self._mu_cache_n = -1

    def _nearest_normalized(self, x: np.ndarray) -> Optional[tuple[float, int]]:
        """(min normalized distance, index). Normalization uses each sigma."""
        if not self._attractors:
            return None
        self._centroid_matrix()
        d = self._mu_cache - x
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        sig = np.maximum(self._sigma_vec, np.finfo(float).eps)
        z = dist / sig
        i = int(np.argmin(z))
        return float(z[i]), i

    def _merge_overlapping(self) -> bool:
        """
        Merge the two attractors whose centroids are closest relative to
        their characteristic scales.

        Returns True if a merge happened.

        No removal by value ranking. If no pair overlaps, nothing is removed:
        exceeding capacity is accepted rather than resolved by ranking.
        """
        n = len(self._attractors)
        if n < 2:
            return False

        mus = np.stack([a.mu for a in self._attractors])
        sig = np.array([a.sigma for a in self._attractors])

        sq = np.sum(mus * mus, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (mus @ mus.T)
        np.maximum(d2, 0.0, out=d2)
        dist = np.sqrt(d2)
        np.fill_diagonal(dist, np.inf)

        radius = np.maximum(sig[:, None], sig[None, :])
        merge_mask = dist < radius
        np.fill_diagonal(merge_mask, False)

        if not merge_mask.any():
            return False

        masked = np.where(merge_mask, dist, np.inf)
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        if i > j:
            i, j = j, i
        a, b = self._attractors[i], self._attractors[j]
        for m in b.members:
            a.members.append(m)
        if len(a.members) > 0:
            a.mu = np.mean(np.stack(a.members), axis=0)
        hi = max(a.amp, b.amp)
        a.amp = hi + (self.ceiling - hi) * self.assign_ratio
        fa = a._decay_factor if a._decay_factor is not None else 1.0
        fb = b._decay_factor if b._decay_factor is not None else 1.0
        a._decay_factor = 0.5 * (fa + fb)
        a._sigma_cache = None
        del self._attractors[j]
        self._invalidate_scale()
        return True

    # Compatibility alias.
    def _prune(self) -> None:
        self._merge_overlapping()

    def _population_visit_rate(self) -> Optional[float]:
        """
        Median visit rate over existing regions, in assignments per step.

        This is the quantity birth depth actually needs: how often a region of
        this structure is reached. The total step count cannot answer it -- it
        says how long the run has lasted, not how busy the structure is. It is
        the same mistake that `_born_step` was introduced to fix in
        viability.py; at the time only the R half was aligned and the `p`
        half was left using the total, which is what made it survive here.

        Returns None when there are too few regions to measure a median.
        """
        if len(self._attractors) < 8:
            return None
        total = max(int(self._total_steps), 1)
        rates = []
        for a in self._attractors:
            life = max(total - int(a._born_step), 1)
            assigned = int(getattr(a, "_assigned", 0))
            if assigned > 0:
                rates.append(assigned / float(life))
        if not rates:
            return None
        rate = float(np.median(rates))
        return rate if rate > 0 else None

    def _birth_decay_factor(self) -> float:
        """Per-region recovery multiplier, drawn deterministically.

        It used to come from self._rng, which is not persisted, so a replay
        could not reproduce a birth. The draw is now a pure function of the
        step index: same history in, same region out.
        """
        r = np.random.default_rng((int(self._total_steps) * 2654435761) % (2 ** 32))
        f = 1.0 + float(r.normal(0.0, self.decay_spread))
        return max(f, 0.05)


    def _birth_depth(self, scale: float) -> float:
        """
        Depth a newly created region starts with.

        Computed the same way carrying_depth computes what a region can hold at
        a given visit rate. The two therefore agree at birth instead of
        differing by a factor of twenty.

        Correction, 2026-11-07 (external audit, docs/2026-11-07_外部评估_连续存在视角.md §3.2)
        -----------------------------------------------------------------------------------
        The visit rate used to be `1.0 / total_steps`, i.e. "one assignment over
        every step this run has ever taken". By that formula birth depth falls
        with elapsed time: with the defaults (assign_ratio 0.05, recover_rate
        0.002, cadence 16, ceiling 1.0, min_amp 1e-3) it is

            3.85e-2 at 1e4 steps, 2.70e-3 at 147664 steps (the longest run on
            record), 1.00e-3 at 399600 steps, 4.00e-4 at 1e6 steps

        and min_amp is 1e-3. So past roughly 400k steps every new region is born
        already below the survival line and is deleted by the next recover():
        the structure stops growing and only deepens what exists. A run that is
        supposed to last cannot have a term that makes new structure impossible
        the longer it lasts.

        The rate therefore comes from the population, measured: the median
        assignments-per-step of the regions that already exist. It does not
        depend on how long the run has lasted. Cold start (fewer than 8 regions,
        nothing to measure) falls back to 1 assignment per 1 step, which is what
        a brand new structure with a single region actually has.
        """
        p = self._population_visit_rate()
        if p is None:
            p = 1.0
        num = p * self.assign_ratio
        every = max(int(self._recover_every_hint), 1)
        R = self.recover_rate / every
        denom = num + R
        if denom <= 0:
            return 0.0
        return float(self.ceiling * num / denom)

    # ---------- folding ----------

    def fold_candidates(self, min_assigned: int = 2, age_factor: float = 8.0) -> list[int]:
        """Regions to collapse: rarely reached and old relative to the half-life.

        Not a value ranking. The criterion is \"never needed again\" (assigned fewer
        than `min_assigned` times) and \"old enough that a return is not expected\"
        (age > age_factor * half_life). Capacity discipline, not mortality: the
        region is collapsed into a summary, not judged unimportant.
        """
        half = self.half_life
        if not (half > 0) or not np.isfinite(half):
            half = float("inf")
        now = int(self._total_steps)
        out: list[int] = []
        for i, a in enumerate(self._attractors):
            if int(getattr(a, "_assigned", 0)) >= int(min_assigned):
                continue
            age = now - int(getattr(a, "_born_step", 0))
            if age > age_factor * half:
                out.append(i)
        return out

    def apply_fold(self, ids: list[int]) -> int:
        """Collapse the given attractor indices into the folded summary."""
        if not ids:
            return 0
        keep = []
        drop = set(int(i) for i in ids)
        for i, a in enumerate(self._attractors):
            if i in drop:
                self._folded.append({
                    "mu": np.round(np.asarray(a.mu, dtype=float), 1),
                    "amp": float(a.amp),
                    "count": int(getattr(a, "_assigned", 0)),
                })
            else:
                keep.append(a)
        removed = len(self._attractors) - len(keep)
        self._attractors = keep
        self._invalidate_scale()
        return removed

    def fold(self, min_assigned: int = 2, age_factor: float = 8.0) -> int:
        """Fold and journal in one step. Returns how many regions were collapsed."""
        ids = self.fold_candidates(min_assigned=min_assigned, age_factor=age_factor)
        if not ids:
            return 0
        removed = self.apply_fold(ids)
        if self.journal is not None:
            self.journal.append_fold(ids)
        return removed


    # ---------- recovery ----------

    @property
    def half_life(self) -> float:
        """
        Steps for a region's depth to halve at the base recovery rate.

        Derived from recover_rate and the recovery cadence, not chosen. Exposed
        so other layers can forget under the same rule instead of inventing a
        second timescale for the same quantity.
        """
        r = float(self.recover_rate)
        every = max(int(self._recover_every_hint), 1)
        if r <= 0 or r >= 1:
            return float("inf")
        return float(np.log(0.5) / np.log(max(1.0 - r, 1e-12))) * every

    def recover(self, rng: np.random.Generator | None = None) -> None:
        """
        Apply recovery to every attractor.

        Each attractor has a fixed decay factor plus a per-call random
        perturbation. Recovery is not conditioned on recent use.

        Rate scaling by observed return interval
        -----------------------------------------
        A region whose typical revisit interval is much longer than the
        recovery half-life loses almost all its depth between visits and can
        never accumulate any. Measured on the real checkpoint: the median
        revisit interval across regions was 47221 steps, while a region sitting
        at rate 0.002 per 16 steps has a half-life of 5540 steps -- a factor of
        8.5. Under those numbers no region except the one the state never left
        could hold depth, and that is exactly what the data showed: one region
        held 95.8% of all assignments and every other region had a median depth
        of 0.011.

        So each region's rate is divided by the ratio between its own return
        interval and the current half-life. A region visited often keeps the
        full rate; a region visited rarely decays in proportion to how rarely it
        is visited. This does not make everything persist: a region visited once
        and never again still fades, just at a rate that puts its half-life on
        the same scale as the interval over which anything could be expected to
        come back.

        The interval is measured, not configured. Regions with too few
        assignments to estimate an interval keep the unscaled rate.
        """
        if rng is None:
            # 2026-11-08: 确定性重放要求同一 total_steps 得到同一抖动。
            # 之前用 default_rng() 无种子，重放结果不可能一致，durable replay 无从谈起。
            rng = np.random.default_rng(self._total_steps)
        r = self.recover_rate
        every = max(int(self._recover_every_hint), 1)
        if self.journal is not None:
            self.journal.append_recover(int(self._total_steps))

        # Current half-life in steps at the base rate.
        if r > 0:
            half = self.half_life
        else:
            half = float("inf")

        total = max(self._total_steps, 1)
        alive: List[Attractor] = []
        for a in self._attractors:
            if a._decay_factor is None:
                a._decay_factor = max(
                    float(1.0 + self._rng.normal(0.0, self.decay_spread)), 0.05
                )

            # Slow the region down if it is revisited far less often than the
            # base half-life would allow.
            rate_scale = 1.0
            if a._assigned >= MIN_ASSIGN_FOR_INTERVAL and half > 0:
                interval = total / float(a._assigned)
                if interval > half:
                    # divide by the ratio, capped so a single stray region does
                    # not end up effectively frozen
                    rate_scale = half / interval
                    rate_scale = max(rate_scale, MIN_RATE_SCALE)

            jitter = 1.0 + self.recover_jitter * float(rng.normal(0.0, 1.0))
            jitter = max(jitter, 0.0)
            rate = r * a._decay_factor * jitter * rate_scale
            a.amp *= max(1.0 - rate, 0.0)
            if a.amp >= self.min_amp:
                alive.append(a)
        if len(alive) != len(self._attractors):
            self._invalidate_scale()
        self._attractors = alive
        self._apply_scale_floor()

    # ---------- observation of structure ----------

    def salient(self, threshold_ratio: float = 0.5) -> List[Attractor]:
        """
        Attractors whose depth is at least `threshold_ratio` of the deepest.

        `threshold_ratio` is an observation parameter, not a mechanism:
        it selects which attractors to report, and does not affect dynamics.
        """
        if not self._attractors:
            return []
        deepest = max(a.amp for a in self._attractors)
        cutoff = deepest * threshold_ratio
        return [a for a in self._attractors if a.amp >= cutoff]

    def depth_profile(self) -> np.ndarray:
        return np.array([a.amp for a in self._attractors], dtype=float)

    def nearest_attractor(self, x: np.ndarray) -> Optional[tuple[int, float]]:
        """Nearest salient attractor, by normalized distance (vectorised)."""
        sals = self.salient()
        if not sals:
            return None
        x = np.asarray(x, dtype=float)
        mus = np.stack([a.mu for a in sals])
        sig = np.maximum(
            np.array([a.sigma for a in sals], dtype=float),
            np.finfo(float).eps,
        )
        d = mus - x
        z = np.sqrt(np.einsum("ij,ij->i", d, d)) / sig
        i = int(np.argmin(z))
        return i, float(z[i])

    # Compatibility alias.
    def nearest_basin(self, x: np.ndarray) -> Optional[tuple[int, float]]:
        return self.nearest_attractor(x)


# Compatibility alias for the previous class name.
Terrain = StateSpace
Trace = Attractor
