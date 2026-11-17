"""
src/L5_cognition/landscape.py

Potential function composed of three terms.

    V(x) = V_state(x) + V_deflect(x) + V_drive(x) + V_self(x)

1. **V_state**   from StateSpace. Formed entirely by observed states.
2. **V_deflect** from L2 (state field). Tilts the potential; adds no minima.
3. **V_drive**   from L4 (unresolved tension). Tilts the potential; adds no minima.
4. **V_self**    from L8 (identity). Weak tilt.

V_deflect and V_drive are separate terms from separate sources and are
summed: they describe different quantities (current state field vs.
unresolved structure) and must not cancel one another.

Zero point
----------

The zero point is the median of warm-up L0 statistics projected into the
L5 space. It is not an origin chosen by the designer.

Former terminology
------------------

This file previously described the potential as "terrain". That is a
spatial metaphor; the object is an energy function on a vector space.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .terrain import StateSpace


@dataclass
class LandscapeState:
    """Current modulation parameters, read by Dynamics."""
    zero_point: np.ndarray
    deflect: Optional[np.ndarray]
    drive: Optional[np.ndarray]
    self_bias: Optional[np.ndarray]
    noise_amplitude: float


class Landscape:
    """Holder of the potential function. Does not advance the state."""

    def __init__(self, dim: int = 128, recover_rate: float = 0.002,
                 max_attractors: int = 8192):
        self.D = int(dim)
        self.state_space = StateSpace(
            dim=self.D, recover_rate=recover_rate, max_attractors=max_attractors
        )

        self._zero: Optional[np.ndarray] = None
        self._deflect: Optional[np.ndarray] = None
        self._drive: Optional[np.ndarray] = None
        self._self_bias: Optional[np.ndarray] = None
        # Global slow direction. Distinct from the per-region rest field:
        # rest releases a region, swing gives the whole space a direction.
        self._swing: Optional[np.ndarray] = None
        # No default. The fluctuation amplitude has to come from upstream; a
        # silent 0.0 here means the potential alone decides the trajectory, the
        # state settles into the nearest minimum, and nothing moves again. That
        # failure looks exactly like a converged system, which is why the
        # default is None rather than zero.
        self._noise_amplitude: Optional[float] = None

    # Compatibility alias: the previous attribute name.
    @property
    def terrain(self) -> StateSpace:
        return self.state_space

    @terrain.setter
    def terrain(self, v: StateSpace) -> None:
        self.state_space = v

    # ---------- zero point ----------

    def calibrate_zero(self, samples: np.ndarray) -> np.ndarray:
        """Zero point = median of the given states."""
        X = np.asarray(samples, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.D:
            raise ValueError(f"samples must be (n, {self.D}), got {X.shape}")
        self._zero = np.median(X, axis=0)
        return self._zero

    @property
    def zero(self) -> np.ndarray:
        if self._zero is None:
            raise RuntimeError("Landscape not calibrated: no zero point")
        return self._zero

    # ---------- modulation ----------

    def set_deflection(self, v: Optional[np.ndarray]) -> None:
        """L2 state-field tilt."""
        self._deflect = None if v is None else np.asarray(v, dtype=float)

    def set_drive(self, v: Optional[np.ndarray]) -> None:
        """L4 unresolved-tension tilt. Summed with set_deflection, not replacing it."""
        self._drive = None if v is None else np.asarray(v, dtype=float)

    def set_self_bias(self, v: Optional[np.ndarray]) -> None:
        """L8 identity tilt."""
        self._self_bias = None if v is None else np.asarray(v, dtype=float)

    def set_swing(self, v: Optional[np.ndarray]) -> None:
        """Global slow-direction tilt. Summed with the others."""
        self._swing = None if v is None else np.asarray(v, dtype=float)

    def set_noise_amplitude(self, a) -> None:
        """
        Fluctuation, supplied by upstream. Never a constant.

        Scalar: isotropic, every direction pushed equally. Vector: directional,
        the fluctuation has a shape. The dynamics handles both; a vector is the
        more informative case because an isotropic push against a deep
        attractor does nothing in particular.
        """
        if a is None:
            self._noise_amplitude = None
            return
        arr = np.asarray(a, dtype=float)
        if arr.ndim == 0:
            self._noise_amplitude = float(max(float(arr), 0.0))
        else:
            self._noise_amplitude = arr.ravel()

    @property
    def noise_amplitude(self):
        """The amplitude. Raises if upstream has not supplied one."""
        if self._noise_amplitude is None:
            raise RuntimeError(
                "noise amplitude not set: the caller must supply it from "
                "upstream before stepping. A default value would make the "
                "system settle silently under the potential alone."
            )
        return self._noise_amplitude

    @property
    def noise_is_set(self) -> bool:
        return self._noise_amplitude is not None

    # ---------- potential and gradient ----------

    def potential(self, x: np.ndarray) -> float:
        z = self.zero
        r = np.asarray(x, dtype=float) - z

        v = self.state_space.potential(x)

        if self._deflect is not None:
            v += float(-self._deflect @ r) * 0.01

        if self._drive is not None:
            v += float(-self._drive @ r) * 0.01

        if self._self_bias is not None:
            v += float(-self._self_bias @ r) * 0.001

        if self._swing is not None:
            v += float(-self._swing @ r) * 0.01

        return float(v)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Analytic gradient of the potential."""
        x = np.asarray(x, dtype=float)
        g = self.state_space.gradient(x)

        if self._deflect is not None:
            g += -self._deflect * 0.01
        if self._drive is not None:
            g += -self._drive * 0.01
        if self._self_bias is not None:
            g += -self._self_bias * 0.001
        if self._swing is not None:
            g += -self._swing * 0.01
        return g

    # ---------- observation and recovery ----------

    def record(self, x: np.ndarray, t_event: str | None = None,
               surprise: float | None = None):
        """Register one observed state in the state space.

        `surprise` 传下去给 StateSpace.record，用来给「她此刻所在的区域」记份量
        （2026-11-17：一个地方让她没预测到过，它就难忘——遗忘曲线的来源）。
        """
        return self.state_space.record(x, t_event, surprise=surprise)

    # Compatibility alias.
    def pass_through(self, x: np.ndarray, t_event: str | None = None):
        return self.state_space.record(x, t_event)

    def recover(self) -> None:
        """Apply recovery. Call frequency is decided by the caller."""
        self.state_space.recover()

    def state(self) -> LandscapeState:
        return LandscapeState(
            zero_point=self.zero,
            deflect=self._deflect,
            drive=self._drive,
            self_bias=self._self_bias,
            noise_amplitude=self._noise_amplitude,
        )
