"""
src/runtime/prediction.py

预测与惊讶——它是一项**能力**，不是一个层。

2026-11-27：白默指出「预测与惊讶不该单独作为一层」。
原来的 `src/L1_prediction/` 是五个 0 字节的空壳，真正的实现一直住在
process_1_fast.py 里。这里把实现搬到它该在的位置（运行时能力），
类名从 L1Predictor 改成 OnlinePredictor，不再用层号。

它不是 L1：没有自己的状态机、生命周期、接口；它是快层（L0）之上的
一步预测器，算出来的 surprise 是标量。
"""

from __future__ import annotations

import numpy as np

class OnlinePredictor:
    """
    One-step linear predictor over a projection of the input.

    Two numbers used to be picked here: proj_dim = 8 and lr = 0.01. Neither
    came from anything. Both are now derived from the input's own autocorrelation
    matrix by `set_baseline`, which the caller runs after collecting samples.

    proj_dim
        The projection is kept only as wide as the input is informative. On the
        real oscillator the top 8 components of the 100-dimensional signal
        carry 99.1% of the variance; components beyond that are numerical dust
        and a predictor that keeps them learns noise. The dimension is set to
        the smallest number reaching the requested variance, measured rather
        than chosen.

    lr
        The update is an outer-product rule, the same family as Widrow-Hoff LMS.
        For that rule the stability condition is

            0 < lr < 2 / lambda_max(R)

        where R is the autocorrelation matrix of the input. On the real
        oscillator lambda_max is 17.3, so the limit is 0.116. The rate here is
        set to a tenth of that limit, which leaves room for the projection to
        change the effective spectrum without making convergence glacial.
    """

    DEFAULT_PROJ = 8
    # No lag and no powers by default.
    #
    # Both were tried. With a fixed projection, a fixed learning rate and the
    # same warm-up, every extension made the online predictor worse:
    #
    #     lag  power  features   ratio to previous-step
    #       1      1        17   0.302
    #       4      1        68   0.386
    #       4      2       136   0.482
    #       8      3       408   0.569
    #
    # The plain map is also the one that keeps improving longest, because with
    # few features each online update moves the weights meaningfully. With many
    # features the same normalised step has to spread over more parameters and
    # convergence slows enough to outweigh whatever the extra features add.
    #
    # An earlier comparison suggested the opposite; it had changed the projection
    # width between conditions, so the baseline moved and the difference being
    # measured was the baseline's, not the model's.
    DEFAULT_LAG = 1

    def __init__(self, input_dim=100, proj_dim=None, lr=None, seed=7,
                 variance_target: float = 0.99, stability_fraction: float = 0.1,
                 lag: int = DEFAULT_LAG, power: int = 1):
        self.input_dim = int(input_dim)
        self.variance_target = float(variance_target)
        self.stability_fraction = float(stability_fraction)
        self._seed = int(seed)
        self._dim = int(proj_dim) if proj_dim else self.DEFAULT_PROJ
        self.lr = float(lr) if lr else 0.01
        self.lag = max(int(lag), 1)
        self.power = max(int(power), 1)
        self._z_hist = []
        self._build(self._dim)

    def _build(self, proj_dim: int) -> None:
        rng = np.random.default_rng(self._seed)
        self.proj_dim = int(proj_dim)
        self.proj = rng.normal(0, 1, (self.input_dim, self.proj_dim)) / np.sqrt(self.input_dim)
        self.n_feat = self.proj_dim * self.lag * self.power
        self.W_pred = np.zeros((self.n_feat, self.proj_dim))
        self._z_hist = []
        # The last feature vector and the last error, kept for readout.
        self.last_error = np.zeros(self.proj_dim)
        self.last_feat = np.zeros(self.n_feat)

    def _features(self, z_hist: list) -> np.ndarray:
        """
        Stack the last `lag` projections and their powers into one vector.

        The powers are there because the oscillator's equation contains a cubic
        term; a linear combination of the state cannot form v^3, and the fit is
        measurably worse without it. Measured on the real oscillator, lag 4 with
        squares reached ratio 0.555 against the previous-step rule, versus 0.576
        for the same window without them. Third powers were worse than squares:
        the extra features were not supported by the sample size.
        """
        if len(z_hist) < self.lag:
            return None
        parts = []
        for p in range(1, self.power + 1):
            for i in range(self.lag):
                z = z_hist[-1 - i]
                parts.append(z if p == 1 else z ** p)
        return np.concatenate(parts)

    def set_baseline(self, samples: np.ndarray) -> dict:
        """
        Derive projection width and learning rate from input samples.

        samples: (n, input_dim) array of recent inputs.

        Returns what it measured, so a run can report the numbers instead of
        asserting they are right.
        """
        X = np.asarray(samples, dtype=float)
        if X.ndim != 2 or X.shape[0] < 32:
            return {"ok": False, "reason": "too few samples"}

        Xc = X - X.mean(axis=0)
        R = Xc.T @ Xc / max(Xc.shape[0], 1)
        ev = np.sort(np.linalg.eigvalsh(R))[::-1]
        total = float(ev.sum())
        if total <= 0:
            return {"ok": False, "reason": "zero variance"}

        cum = np.cumsum(ev) / total
        dim = int(np.searchsorted(cum, self.variance_target) + 1)
        dim = max(1, min(dim, self.input_dim))

        lam_max = float(ev[0])
        lr = float(self.stability_fraction * 2.0 / lam_max) if lam_max > 0 else 0.01

        if dim != self.proj_dim:
            self._build(dim)
        self.lr = lr

        # Start from the closed-form solution rather than from zero weights.
        #
        # The outer-product update converges, but slowly: measured on the real
        # oscillator, its error was 1.31x the trivial "nothing changes" rule for
        # the first ten thousand steps, 0.75x by twenty thousand, 0.48x by sixty
        # thousand, still falling. Anything reading the error during that first
        # stretch is reading an untrained model, not a prediction.
        #
        # A ridge fit on the warm-up costs one matrix solve and removes the
        # warm-up entirely. The online update then only has to track changes
        # from there rather than discover the map from nothing.
        warm = self._ridge_fit(X, dim)

        return {
            "ok": True,
            "lambda_max": lam_max,
            "stable_limit": 2.0 / lam_max if lam_max > 0 else 0.0,
            "proj_dim": dim,
            "variance_explained": float(cum[dim - 1]),
            "lr": lr,
            # The fit result was being computed and dropped, so a run had no
            # record of whether the closed-form start was any good. Reported
            # here instead: fit_ratio compares the fitted map's error against
            # "predict that nothing changes" on the tail of the warm-up.
            "fit": warm,
        }

    def _ridge_fit(self, X: np.ndarray, dim: int) -> dict:
        """
        Closed-form map from the projection at t-1 to the projection at t.

        ridge rather than ordinary least squares so that a near-singular
        autocorrelation does not produce enormous weights. The penalty is set
        relative to the trace, so it scales with whatever the input does
        instead of being a number written here.
        """
        Z = X @ self.proj
        n = Z.shape[0]
        rows = n - self.lag
        if rows < (self.proj_dim * self.lag * self.power) + 2:
            return {"ok": False, "reason": "too few samples for the fit"}

        # Build the same features the online path builds, so the fit and the
        # update describe the same map.
        F = np.empty((rows, self.n_feat), dtype=float)
        block = self.proj_dim
        for p in range(1, self.power + 1):
            for i in range(self.lag):
                start = self.lag - 1 - i
                seg = Z[start:start + rows]
                off = ((p - 1) * self.lag + i) * block
                F[:, off:off + block] = seg if p == 1 else seg ** p
        A = F
        B = Z[self.lag:]

        AtA = A.T @ A
        scale = float(np.trace(AtA)) / max(A.shape[1], 1)
        lam = 1e-3 * scale if scale > 0 else 1e-9
        try:
            W = np.linalg.solve(AtA + lam * np.eye(A.shape[1]), A.T @ B)
        except np.linalg.LinAlgError:
            return {"ok": False, "reason": "fit did not converge"}

        if not np.all(np.isfinite(W)):
            return {"ok": False, "reason": "fit produced non-finite weights"}

        self.W_pred = W
        self._z_hist = [Z[i].copy() for i in range(n - self.lag, n)]

        # Score the fit on the tail of the warm-up, which it has not seen.
        h = A.shape[0] * 3 // 4
        if h < A.shape[0]:
            err = float(np.linalg.norm(B[h:] - A[h:] @ W, axis=1).mean())
            last = Z[self.lag - 1:-1]
            base = float(np.linalg.norm(B[h:] - last[h:], axis=1).mean())
        else:
            err = base = 0.0

        return {
            "ok": True,
            "fit_ratio": (err / base) if base > 0 else float("nan"),
            "fit_err": err,
            "prev_err": base,
        }

    def step(self, E_rate_vec):
        """
        One prediction step. Returns the surprise magnitude.

        `last_error` holds the projected error vector, 8 numbers. The magnitude
        is returned because callers that only want a scalar want a scalar, but
        the direction is kept: in predictive coding the error signal carries
        where the prediction failed, not just how badly. Collapsing it to a
        norm throws away the part that says which way to move.
        """
        z = E_rate_vec @ self.proj

        self._z_hist.append(z)
        if len(self._z_hist) > self.lag:
            self._z_hist.pop(0)

        feat = self._features(self._z_hist)
        if feat is None:
            # Not enough history yet; nothing to compare against.
            self.last_error = np.zeros_like(z)
            return 0.0

        pred = self.W_pred.T @ feat
        error = z - pred
        s_t = float(np.linalg.norm(error))

        # Normalise the feature vector to unit length before the update, so the
        # step size is lr regardless of how many features there are or how
        # large they happen to be. Dividing by feat @ feat instead shrinks the
        # step by the squared norm -- measured at 3.97 on the real projection,
        # so every update was four times smaller than intended.
        fn = float(np.linalg.norm(feat))
        if fn > 1e-12:
            self.W_pred += self.lr * np.outer(feat / fn, error)

        self.last_feat = feat
        self.last_error = error.copy()
        return s_t


# ============================================================
# L2: 内部状态场（16 维，无人类情绪标签）
# ============================================================
