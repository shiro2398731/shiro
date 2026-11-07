"""
src/L5_cognition/noise.py

Turns the prediction error into a fluctuation.

Two things the error carries
----------------------------

A prediction error is a vector, not a number. The magnitude says the prediction
was wrong by this much; the direction says which way it was wrong. Predictive
coding uses both -- precision weighting modulates the magnitude, and the
direction is what drives inference.

An earlier version returned only the norm. Two consequences followed:

  * the magnitudes spanned three orders of magnitude (p1 = 0.0006, p99 = 0.371
    on the real L0/L1 pair), so using it directly made the state jump;
  * all of that enormous variation carried no direction. A 128-dimensional
    fluctuation built from a scalar is isotropic, which means it pushes every
    direction equally and therefore toward nowhere.

What is done here
-----------------

The error vector is projected into the space the dynamics live in, using the
same projection the predictor uses, run backwards. The projection from 100 to 8
dimensions is `z = x @ P`; the corresponding map from 8 back into 100 is
`e @ P.T`, which places an 8-dimensional error back along the axes it came from.

The scalar magnitudes are then divided by their own median so that a typical
step has amplitude comparable to the structure's own gradient. That is a
normalisation of the series by itself, not a scale chosen here, and it is
reported so it can be checked.

Why not simply cap the magnitude
--------------------------------

Capping alone would leave the fluctuation isotropic, and an isotropic push
against a deep attractor does nothing. The direction is the part that matters.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class ErrorNoise:
    """
    Fluctuation derived from the prediction error vector.

    dim       : dimension of the space the dynamics live in (128)
    window    : history length defining the typical magnitude
    gain      : multiplier on the normalised amplitude. Left at 1.0 the
                fluctuation has the same magnitude as a typical error; this is
                the one number here that is still a choice, and it is reported.
    """

    def __init__(self, dim: int = 128, window: int = 512, gain: float = 1.0):
        self.D = int(dim)
        self.window = int(window)
        self.gain = float(gain)
        self._mag: deque = deque(maxlen=self.window)
        self._last_vec: Optional[np.ndarray] = None
        self._last_norm: float = 0.0
        self._dir_hist: deque = deque(maxlen=self.window)

    def observe(self, error_vec: np.ndarray, proj: np.ndarray) -> np.ndarray:
        """
        Record one prediction error and return the fluctuation vector.

        error_vec : (k,) projected error
        proj      : (100, k) the predictor's projection, used backwards
        """
        e = np.asarray(error_vec, dtype=float).ravel()
        if e.size == 0 or proj.size == 0:
            out = np.zeros(self.D)
            self._last_vec = out
            return out

        mag = float(np.linalg.norm(e))
        if not np.isfinite(mag):
            return self._last_vec if self._last_vec is not None else np.zeros(self.D)
        self._mag.append(mag)

        # Direction in the full space: place the error back along the axes it
        # was projected from. This is the adjoint, not an invention.
        unit = e / mag if mag > 1e-12 else np.zeros_like(e)
        full_in = np.asarray(proj, dtype=float) @ unit
        # The predictor works in L0's 100 dimensions; the state space is 128.
        # The extra components have no corresponding prediction error, so they
        # stay at zero. Padding with noise would put motion into directions the
        # error says nothing about.
        full = np.zeros(self.D)
        m = min(full_in.size, self.D)
        full[:m] = full_in[:m]
        n = float(np.linalg.norm(full))
        if n <= 1e-12:
            out = np.zeros(self.D)
            self._last_vec = out
            return out
        full = full / n

        # Magnitude, normalised by the typical error so far. A run whose errors
        # are all small moves gently; one whose errors are large moves more.
        if len(self._mag) >= 32:
            arr = np.asarray(self._mag, dtype=float)
            ref = float(np.median(arr))
        else:
            ref = mag
        amp = (mag / ref) if ref > 1e-12 else 0.0
        # One step can carry an error far above the recent level; the cap is a
        # high quantile of the series itself, not a constant.
        if len(self._mag) >= 32:
            hi = float(np.percentile(np.asarray(self._mag, dtype=float), 99))
            if hi > 1e-12:
                amp = min(amp, hi / ref)

        out = full * amp * self.gain
        self._last_vec = out
        self._last_norm = float(np.linalg.norm(out))
        self._dir_hist.append(e / mag if mag > 1e-12 else np.zeros_like(e))
        return out

    @property
    def last(self) -> Optional[np.ndarray]:
        return None if self._last_vec is None else self._last_vec.copy()

    @property
    def last_norm(self) -> float:
        return self._last_norm

    def describe(self) -> dict:
        if not self._mag:
            return {"n": 0, "norm": 0.0}
        arr = np.asarray(self._mag, dtype=float)
        return {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "p99": float(np.percentile(arr, 99)),
            "norm": self._last_norm,
        }
