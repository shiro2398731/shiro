"""
src/L5_cognition/dynamics.py

Trajectory of the state vector under the potential.

Equations
---------

    dx/dt = v
    dv/dt = -gamma * v - grad V(x) + xi(t)

- gamma : damping coefficient. Determines how far the state travels before
          the potential term dominates. Large gamma approaches gradient
          descent (immediate convergence, no escape); small gamma overshoots
          and no attractor forms.
- grad V: supplied by Landscape.
- xi(t): fluctuation term. Its amplitude is not defined here; it is supplied
         by upstream through Landscape.set_noise_amplitude().

On inertia
----------

With gradient terms alone the state converges to the nearest local minimum
and remains there under arbitrarily small perturbation. Inertia allows the
state to cross shallow attractors and to be retained by deep ones.

On dt
-----

dt is the numerical integration step. It is a property of the integrator,
not of the system being modelled. It drives nothing; the caller decides
how often step() is invoked.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .landscape import Landscape


@dataclass
class DynamicState:
    """Position and velocity. The complete dynamical state of L5."""
    x: np.ndarray            # (D,)
    v: np.ndarray            # (D,)

    def copy(self) -> "DynamicState":
        return DynamicState(x=self.x.copy(), v=self.v.copy())


class Dynamics:
    """
    Semi-implicit Euler integrator.

    Semi-implicit rather than explicit: with explicit Euler the energy
    diverges on this class of damped potential fields.
    """

    def __init__(self, dim: int = 128, gamma: float = 0.05, dt: float = 0.01):
        """
        gamma: damping coefficient
        dt   : integration step
        """
        self.D = int(dim)
        self.gamma = float(gamma)
        self.dt = float(dt)

    def init_state(self, zero: np.ndarray, seed: int = 0) -> DynamicState:
        """
        Initial position = zero point (from warm-up L0 statistics),
        initial velocity = 0.

        A random initial position would also be a designer-chosen value.
        """
        x = np.asarray(zero, dtype=float).copy()
        v = np.zeros(self.D, dtype=float)
        return DynamicState(x=x, v=v)

    def step(
        self,
        st: DynamicState,
        landscape: Landscape,
        rng: np.random.Generator,
    ) -> DynamicState:
        """Advance one integration step. Returns a new DynamicState."""
        sigma = landscape.noise_amplitude
        grad = landscape.gradient(st.x)

        # A scalar amplitude means an isotropic push; a vector means the
        # fluctuation has a direction. See Landscape.set_noise_amplitude.
        if np.ndim(sigma) == 0:
            noise = rng.normal(0.0, 1.0, self.D) * float(sigma)
        else:
            noise = np.asarray(sigma, dtype=float)
            if noise.shape[0] != self.D:
                raise ValueError(
                    f"noise vector has {noise.shape[0]} components, "
                    f"state has {self.D}"
                )
        dv = -self.gamma * st.v - grad + noise
        v_new = st.v + dv * self.dt
        x_new = st.x + v_new * self.dt

        return DynamicState(x=x_new, v=v_new)

    # ---------- readouts ----------

    @staticmethod
    def speed(st: DynamicState) -> float:
        return float(np.linalg.norm(st.v))

    @staticmethod
    def displacement(st: DynamicState, x_ref: np.ndarray) -> float:
        return float(np.linalg.norm(st.x - x_ref))
