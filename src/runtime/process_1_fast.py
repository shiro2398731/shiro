"""
src/runtime/process_1_fast.py — P1: L0 + L1 + L2

L0: E/I FHN 网络（快层闭合，I 滞后 E）
L1: 预测与惊讶（s_t = 预测误差，在线线性预测器）
L2: 内部状态场（16 维连续向量，无人类情绪标签）

L2 不输出 AffectTag，不写 "curious"/"uneasy"/"bored"。
每一维是一个可计算的功能性量。偏差向量由 L5 自己赋予意义。
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

import collections
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.common.bus import Bus
from src.common.observe import Observer
from src.common.types import EnvelopeEvent, PhaseBroadcast, StateVector, SurpriseSignal, TextSignalEvent, safe_from_json
import src.common.persist as persist


def _sigmoid(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * np.clip(x, -10.0, 10.0)))


# ============================================================
# L0: E/I FHN
# ============================================================

class SlowAccumulator:
    """
    慢累积量：一个只累积、只清除、不自振荡的积分器。

    设计约束（来自白默 2026-11-02）：
      代码里不得出现 sleep / wake / awake / 疲劳 等词。
      “清醒”这个词目前没有指称物——没有主体的时候，
      “她醒着吗”是个问不出意义的问题。只描述它是什么，不描述它意味着什么。

    动力学：

        dP/dt = (fast - ref) / tau_up      当 fast > ref
        dP/dt = (fast - ref) / tau_down    当 fast < ref

    ref 是 fast 自身历史的中位数，**不是写死的 0.5**。
    连续累积，不做“高于阈值/低于阈值”的二分判断——因为 fast 在 up/down 之间
    大致均匀分配时间，任何二分写法都会得到一半一半的平庸结果。

    tau_down > tau_up 是不对称性，但这不是“清醒vs睡眠”，只是“涨得快、落得慢”
    这个纯动力学性质。

    关于 suppression_gate：
      这是留给 L4（动机）的一个空位。当 L4 存在后，它可以把这个系数
      从 1.0 压到较低值。现在没有人调用它，它恒为 1.0。

      【待接入】调用者：src/L4_drive/deflection.py（尚未实现）
    """

    HIST = 200          # fast 历史窗口（步数），用于算 ref

    def __init__(self, tau_up_s: float, tau_down_s: float, dt_s: float):
        self.tau_up = float(tau_up_s)
        self.tau_down = float(tau_down_s)
        self.dt = float(dt_s)
        self.P = 0.0
        self._hist = collections.deque(maxlen=self.HIST)
        self.suppression_gate = 1.0
        # The reference is not the median. With tau_up < tau_down, a reference
        # at the median leaves a net positive drift, because the same distance
        # above counts for more than the same distance below. Measured on the
        # oscillator: 53% of steps above its own median, and the value climbed
        # monotonically to 0.016 over 400000 steps without ever turning over.
        # A quantity that only ever rises with elapsed time carries no
        # information about state, only about how long the run has lasted.
        #
        # So the reference is adjusted until the two sides balance. The balance
        # condition is
        #
        #     P(above) * E[d | d>0] / tau_up  =  P(below) * E[-d | d<0] / tau_down
        #
        # and the reference is moved by a small step in the direction that
        # reduces the imbalance, so it settles wherever this oscillator's own
        # asymmetry puts it rather than at a number written here.
        self._ref_shift = 0.0
        self._ref_rate = 0.02
        # Strength of the decay toward the recent level. At 1.0 the decay has a
        # time constant of one history window; a larger value lets the
        # integrator's drift dominate again.
        self.decay_gain = 1.0

    def step(self, fast: float) -> float:
        self._hist.append(fast)
        if len(self._hist) < 8:
            return self.P

        hist = np.asarray(self._hist, dtype=float)

        ref = self._balanced_ref(hist)
        delta = fast - ref
        tau = self.tau_up if delta > 0 else self.tau_down
        drive = self.dt * delta / tau

        # Without a decay term P is a pure integrator on a signal whose rises
        # and falls are not equal in magnitude: fast climbs sharply and falls
        # slowly, so the sum of positive contributions exceeds the sum of
        # negative ones and P rises forever. Measured: 0.016 after 400000 steps
        # and still climbing linearly. Clipping at 1.0 is not a decay, it is a
        # cut, and it only takes effect after the drift has already made the
        # quantity meaningless.
        #
        # So a decay proportion is added, pulling P toward its own recent
        # median. The target is not a constant: it is whatever level the
        # accumulated history says this oscillator has been sitting at. The
        # decay rate is one over the history length, so P follows its own
        # history on the same timescale the history is kept, and no separate
        # timescale has to be invented.
        target = float(np.median(hist))
        decay = (target - self.P) / max(float(self.HIST), 1.0) * self.decay_gain

        self.P = float(np.clip(self.P + drive + decay, 0.0, 1.0))
        return self.P

    def _balanced_ref(self, hist: np.ndarray) -> float:
        """
        The value r for which the expected change of P is zero.

        Computed by weighting the window: above-r mass divided by tau_up must
        equal below-r mass divided by tau_down. A bisection over the sorted
        window finds it without assuming any distribution shape.
        """
        x = np.sort(hist)
        n = x.size
        lo, hi = 0, n - 1
        target = self.tau_up / self.tau_down
        for _ in range(40):
            mid = (lo + hi) // 2
            r = x[mid]
            above = x[x > r]
            below = x[x < r]
            if above.size == 0 or below.size == 0:
                break
            up_mass = float(np.abs(above - r).sum())
            dn_mass = float(np.abs(below - r).sum())
            # up_mass / tau_up  ?  dn_mass / tau_down
            # -> up_mass * tau_down  ?  dn_mass * tau_up
            if up_mass * self.tau_down > dn_mass * self.tau_up:
                lo = mid + 1
            else:
                hi = mid - 1
            if lo > hi:
                break
        idx = min(max((lo + hi) // 2, 0), n - 1)
        _ = target
        return float(x[idx])

    @property
    def expressed(self) -> float:
        return self.P * self.suppression_gate


class FHNNetwork:
    def __init__(self, seed: int = 42, tau_up_s: float = 3600.0, tau_down_s: float = 7200.0, dt_s: float = 0.001):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.step_count = 0
        self.a = 0.7
        self.b = 0.8

        self.E_n = 100
        self.E_eps = 0.08
        self.E_I_base = 0.40

        # External drive state for the input path (see _external_drive).
        # _ext_pattern is the shape applied to the population; _ext_amp is how
        # strongly it acts right now. Both are set from outside by main().
        self._ext_pattern = np.zeros(self.E_n)
        self._ext_amp = 0.0
        self._ext_decay = 0.0        # set below once dt is known
        self.E_v = rng.uniform(-1.5, 0.5, self.E_n)
        self.E_w = rng.uniform(-0.3, 0.3, self.E_n)
        self.dt = 0.1
        # Decay per fast step for an external signal. Set from the oscillator's
        # own period rather than from a step count: a sentence should bias a
        # few cycles, not a few hundred steps, and the two are the same only by
        # accident. See set_perturbation_span().
        self._ext_decay = 0.98
        self._period_steps = 347.0        # filled in by set_perturbation_span
        self._ext_span_frac = 0.1
        self._fast_std = 1.0              # filled in by set_baseline
        self._input_gain = 1.0
        self._ext_span_steps = 34.7
        mask_EE = rng.random((self.E_n, self.E_n)) < 0.15
        np.fill_diagonal(mask_EE, False)
        self.EE = mask_EE * rng.random((self.E_n, self.E_n)) * 0.15

        self.I_n = 20
        self.I_eps = 0.02
        self.I_I_base = 0.35
        self.I_v = rng.uniform(-1.5, 0.5, self.I_n)
        self.I_w = rng.uniform(-0.3, 0.3, self.I_n)
        mask_EI = rng.random((self.E_n, self.I_n)) < 0.20
        self.EI = mask_EI * rng.random((self.E_n, self.I_n)) * 0.15
        mask_IE = rng.random((self.I_n, self.E_n)) < 0.30
        self.IE = mask_IE * rng.random((self.I_n, self.E_n)) * 0.20
        self.mid_n = 20
        self.mid_eps = 0.008
        self.mid_I_base = 0.10
        self.mid_v = rng.uniform(-1.0, 0.0, self.mid_n)
        self.mid_w = rng.uniform(-0.3, 0.3, self.mid_n)
        self.mid_dt = 1.0
        fm_mask = rng.random((self.E_n, self.mid_n)) < 0.3
        self.E_to_mid = fm_mask * rng.random((self.E_n, self.mid_n)) * 0.3

        self.slow_n = 5
        self.slow_eps = 0.0008
        self.slow_I_base = 0.10
        self.slow_v = rng.uniform(-1.0, 0.0, self.slow_n)
        self.slow_w = rng.uniform(-0.3, 0.3, self.slow_n)
        # dt 是“仿真时间步长”，不是“更新频率”。两层都用同一个 dt 量级（0.1），
        # 慢只是因为 eps 小（0.0008 vs 快层 0.08），而不是因为推得少。
        # 历史教训：
        #   slow_dt=10.0   → dt/eps = 12500，冲过极限环，退化成方波
        #   slow_dt=0.01 + 每 1000 步推一次 → 推进率比快层慢 10000 倍，几乎静止
        # 现在：dt 与中层一致（dt/eps = 125），每步都推。
        self.slow_dt = 0.1
        ms_mask = rng.random((self.mid_n, self.slow_n)) < 0.5
        self.mid_to_slow = ms_mask * rng.random((self.mid_n, self.slow_n)) * 0.05

        self._last_mid = 0.0
        self._last_slow = 0.0

        # Turnover of connections: what wears out is replaced, so the
        # substrate is continuously renewed without changing shape. This is
        # metabolism, not learning -- see src/L0_substrate/metabolism.py.
        from src.L0_substrate.metabolism import Turnover
        self.turnover = Turnover()
        self.turnover.attach(self._coupling_matrices())

        # 慢累积量：无名字的积分器，意义留给后面长
        self.accumulator = SlowAccumulator(
            tau_up_s=tau_up_s,
            tau_down_s=tau_down_s,
            dt_s=dt_s,
        )

    def _step_fast(self, gm):
        E_rate = _sigmoid(self.E_v)
        I_rate = _sigmoid(self.I_v)
        E_coupling = self.EE @ E_rate
        E_inhibition = self.IE.T @ I_rate
        noise = self.rng.normal(0.0, 0.02, self.E_n)
        # External drive, added to the excitatory current.
        #
        # This is the only path by which anything outside the oscillator can
        # affect it. The term is zero unless something was published on
        # l6.text.signal, and it decays on its own so that a single sentence
        # does not hold the network in a permanently shifted state.
        #
        # It is added as current rather than as a state assignment: the
        # oscillator keeps its own dynamics, and the input merely biases them.
        # Setting E_v directly would make the input an instruction rather than
        # a perturbation.
        ext = self._external_drive()
        I_E = self.E_I_base * gm + E_coupling - E_inhibition + noise + ext

        dw_E = self.E_eps * (self.E_v + self.a - self.b * self.E_w)
        self.E_w = np.clip(self.E_w + dw_E * self.dt, -2.0, 2.0)
        dv_E = self.E_v - self.E_v ** 3 / 3.0 - self.E_w + I_E
        self.E_v = np.clip(self.E_v + dv_E * self.dt, -3.0, 3.0)

        E_drive = self.EI.T @ E_rate
        noise_I = self.rng.normal(0.0, 0.02, self.I_n)
        I_I = self.I_I_base * gm + E_drive + noise_I
        dw_I = self.I_eps * (self.I_v + self.a - self.b * self.I_w)
        self.I_w = np.clip(self.I_w + dw_I * self.dt, -2.0, 2.0)
        dv_I = self.I_v - self.I_v ** 3 / 3.0 - self.I_w + I_I
        self.I_v = np.clip(self.I_v + dv_I * self.dt, -3.0, 3.0)

        return float(E_rate.mean()), float(I_rate.mean())

    def _step_mid(self):
        E_rate = _sigmoid(self.E_v)
        drive = self.E_to_mid.T @ E_rate
        noise = self.rng.normal(0.0, 0.02, self.mid_n)
        I = self.mid_I_base + drive + noise
        dw = self.mid_eps * (self.mid_v + self.a - self.b * self.mid_w)
        self.mid_w = np.clip(self.mid_w + dw * self.mid_dt, -2.0, 2.0)
        dv = self.mid_v - self.mid_v ** 3 / 3.0 - self.mid_w + I
        self.mid_v = np.clip(self.mid_v + dv * self.mid_dt, -3.0, 3.0)
        return float(_sigmoid(self.mid_v).mean())

    def _step_slow(self):
        mid_rate = _sigmoid(self.mid_v)
        drive = self.mid_to_slow.T @ mid_rate
        noise = self.rng.normal(0.0, 0.02, self.slow_n)
        I = self.slow_I_base + drive + noise
        dw = self.slow_eps * (self.slow_v + self.a - self.b * self.slow_w)
        self.slow_w = np.clip(self.slow_w + dw * self.slow_dt, -2.0, 2.0)
        dv = self.slow_v - self.slow_v ** 3 / 3.0 - self.slow_w + I
        self.slow_v = np.clip(self.slow_v + dv * self.slow_dt, -3.0, 3.0)
        return float(_sigmoid(self.slow_v).mean())

    def set_baseline(self, period_steps: float, fast_std: float) -> None:
        """
        Set the scales the external input is measured against.

        Both come from the oscillator itself, measured, not chosen:

        period_steps : the oscillator's own period, in fast steps. How long an
                       input lingers is expressed as a fraction of one cycle,
                       because a cycle is the only interval the layer itself
                       defines.
        fast_std     : the oscillator's own fluctuation width. How strong an
                       input is becomes a fraction of that width, so "strong"
                       means the same thing here as it does everywhere else in
                       the layer.
        """
        if period_steps and period_steps > 0:
            self._period_steps = float(period_steps)
        if fast_std and fast_std > 0:
            self._fast_std = float(fast_std)
        self._apply_input_scales()

    def set_perturbation_span(self, fraction: float) -> None:
        """How long an external input lingers, as a fraction of one period."""
        self._ext_span_frac = max(float(fraction), 1e-6)
        self._apply_input_scales()

    def _apply_input_scales(self) -> None:
        """
        Recompute decay and gain from the baselines.

        decay: the residual amplitude after `span` steps should be one percent,
        so decay = 0.01 ** (1 / span).

        gain : an input at full scale moves the network by one standard
        deviation of its own fluctuation. Anything else is a number nobody
        derived.
        """
        span = max(self._period_steps * self._ext_span_frac, 1.0)
        self._ext_span_steps = span
        self._ext_decay = float(0.01 ** (1.0 / span))
        self._input_gain = float(getattr(self, "_fast_std", 1.0))

    def receive(self, vector, gain: float | None = None) -> None:
        """
        Apply an external signal as input current.

        The vector is mapped to the population size and normalized, so any input
        dimension works and the amplitude does not depend on how long the vector
        is. `gain` sets how strong a single sentence can be relative to the
        oscillator's own drive; it is a scale, not a threshold.

        The amplitude decays on its own (see _external_drive), so the network is
        perturbed and then returns to its own dynamics. Nothing here holds a
        state: the input biases the oscillator, it does not set it.
        """
        v = np.asarray(vector, dtype=float).ravel()
        if v.size == 0:
            return
        n = self.E_n
        idx = np.floor(np.linspace(0.0, 1.0, n, endpoint=False) * v.size).astype(int)
        p = v[np.mod(idx, v.size)]
        norm = float(np.linalg.norm(p))
        if norm > 0:
            p = p / norm * np.sqrt(n)
        self._ext_pattern = p
        # Default gain is the oscillator's own fluctuation width, so a signal
        # at unit scale perturbs it by one standard deviation of what it does
        # anyway.
        self._ext_amp = float(self._input_gain if gain is None else gain)

    def _external_drive(self):
        """Current external drive, decaying since the last received signal."""
        if self._ext_amp <= 1e-12:
            return 0.0
        out = self._ext_pattern * self._ext_amp
        self._ext_amp *= self._ext_decay
        return out

    def _coupling_matrices(self) -> dict:
        """The matrices that carry structure, as opposed to state."""
        return {
            "EE": self.EE,
            "EI": self.EI,
            "IE": self.IE,
            "E_to_mid": self.E_to_mid,
            "mid_to_slow": self.mid_to_slow,
        }

    def metabolize(self, activity_scale: float | None = None) -> None:
        """
        Replace worn-out connections by one step's worth.

        activity_scale sets how fast turnover runs. The caller supplies it,
        meant to follow the oscillator's own activity so that more is replaced
        while more is happening. None restores the base rate.

        Nothing here strengthens anything. The number of connections and their
        weight distribution are unchanged by turnover; only which particular
        connections exist changes.
        """
        self.turnover.set_rate_scale(activity_scale)
        self.turnover.step(self._coupling_matrices())

    def step(self, gm=1.0):
        E_rate, I_rate = self._step_fast(gm)        # 慢累积量推进。它读取快层活动，但不反回去改变 L0 的振荡。
        pressure = self.accumulator.step(E_rate)

        if self.step_count % 10 == 0:
            self._last_mid = self._step_mid()
        # 慢层每步都推进（它的“慢”来自 eps，不是来自推得少）
        self._last_slow = self._step_slow()
        self.step_count += 1
        return {
            "step": self.step_count,
            "fast": E_rate,
            "I_rate": I_rate,
            "mid": self._last_mid,
            "slow": self._last_slow,
            "pressure": pressure,
            "E_rate_vec": _sigmoid(self.E_v),
        }


# ============================================================
# L0 包络检测器：在慢包络变量 A 上读相位与周期
# ============================================================

class EnvelopeDetector:
    """
    在 L0 的慢包络变量 A（分钟级）上读相位与周期。

    **重要教训（2026-11-02）**：本类曾接在 fast 瞬时值上，结果测出 0.6 秒的
    毫秒级抖动。根因有两层：
      1. 帧率混淆：把日志的 2 秒采样当成信号自身的时间尺度
      2. 窗口混淆：HIST=64 个采样点 @ 0.001s = 0.064s 的极短窗口

    现在修正：
      - 输入量是 A（慢包络），不是 fast
      - 窗口是**时间窗口**（HIST_WINDOW_S），不是采样点数
      - 采样步长是 SAMPLE_INTERVAL_S，与主循环步长解耦

    阈值：用 A 自身时间窗口的中位数作分界，不是写死 0.5。
    周期：最近 MAX_GAPS 次上穿间隔的中位数。

    周期统计（2026-11-02 新增）：每个完整周期结束时，输出该周期内的
    fast 均值、峰谷差、up 停留比例。这是因为 L2 需要“周期尺度”的量，
    而相邻步差之类的东西只是在测数值抖动。
    """

    SAMPLE_INTERVAL_S = 0.02    # 检测器采样间隔（工程标度，B 类）
                                # 依据：实测 fast 极限环周期约 0.35 秒，
                                # 按采样定理至少需 0.175 秒，留 8 倍余量用 0.02 秒。
    HIST_WINDOW_S = 2.0         # 历史分界参考的时间窗口（约 5-6 个周期）
    MAX_GAPS = 8

    def __init__(self):
        self._hist_t = collections.deque()
        self._hist_v = collections.deque()
        self._gaps = collections.deque(maxlen=self.MAX_GAPS)
        self._last_sample_t = None
        self._was_above = None
        self._last_cross_t = None
        self._phase = "down"

        # 当前周期内的累积统计（每次上穿时清零并结算）
        self._cycle_fasts: list = []
        self._cycle_n_up = 0
        self._cycle_n = 0
        self.last_cycle_stats: dict | None = None

    def _settle_cycle(self) -> None:
        if self._cycle_n > 0:
            arr = np.asarray(self._cycle_fasts, dtype=float)
            self.last_cycle_stats = {
                "fast_mean": float(arr.mean()),
                "amplitude": float(arr.max() - arr.min()),
                "up_ratio": float(self._cycle_n_up) / float(self._cycle_n),
            }
        self._cycle_fasts = []
        self._cycle_n_up = 0
        self._cycle_n = 0

    def update(self, A: float, t: float) -> tuple[str, float | None]:
        """返回 (phase, period_seconds)。周期在样本不足时为 None。"""
        # 按时间采样，不按主循环步数
        if self._last_sample_t is not None and (t - self._last_sample_t) < self.SAMPLE_INTERVAL_S:
            return self._phase, self._period()
        self._last_sample_t = t

        self._hist_t.append(t)
        self._hist_v.append(A)
        while self._hist_t and (t - self._hist_t[0]) > self.HIST_WINDOW_S:
            self._hist_t.popleft()
            self._hist_v.popleft()

        # 样本不足以建立分界时，不停判断
        if len(self._hist_v) < 8:
            return self._phase, self._period()

        ref = float(np.median(self._hist_v))
        above = A >= ref

        if self._was_above is None:
            self._was_above = above
            return self._phase, self._period()

        if above and not self._was_above:
            if self._last_cross_t is not None:
                gap = t - self._last_cross_t
                if gap > 0.0:
                    self._gaps.append(gap)
            self._settle_cycle()      # 上穿 = 一个新周期的开始，结算旧周期
            self._last_cross_t = t
            self._phase = "up"
        elif (not above) and self._was_above:
            self._phase = "down"

        # 累积本周期的统计
        self._cycle_fasts.append(A)
        self._cycle_n += 1
        if above:
            self._cycle_n_up += 1

        self._was_above = above
        return self._phase, self._period()

    def _period(self) -> float | None:
        if not self._gaps:
            return None
        return float(np.median(self._gaps))

    @property
    def period_samples(self) -> int:
        return len(self._gaps)


# ============================================================
# L1: 预测与惊讶
# ============================================================

class L1Predictor:
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

class StalenessGate:
    """
    ARCHITECTURE.md §3.3 的自适应更新判据。

        staleness = |x_now - x_last_computed| / scale(x_self)

    scale 是该量自身在近期历史中的波动尺度（中位绝对偏差 MAD），
    **由实测得到，不是配置常量**。

    含义：当输入偏离它平常波动幅度的程度超过平常幅度本身时，就重算。

    **关键语义**：_last 是“上次真正重算时的值”，不是“上一步的值”。
    如果每步都更新 _last，staleness 就退化成“步间变化”，
    而连续信号的步间变化总是很小，系统将几乎永不重算。（2026-11-02 踩过）

    为什么这样能正确工作：
      - 噪声大的通道 scale 大，自然触发得少
      - 稳定的通道 scale 小，一点变化就能触发
      - 整体变活跃时全体 scale 都变大，更新率同时上浮
    """

    HIST = 256
    MIN_HIST = 16

    def __init__(self, threshold: float = 1.0):
        self.threshold = float(threshold)
        self._hist: dict[str, collections.deque] = {}
        self._last: dict[str, float] = {}

    def _scale(self, name: str) -> float:
        h = self._hist.get(name)
        if h is None or len(h) < self.MIN_HIST:
            return None
        arr = np.asarray(h, dtype=float)
        mad = float(np.median(np.abs(arr - np.median(arr))))
        # MAD 可能为 0（常量输入），此时该通道不驱动更新
        return mad if mad > 1e-9 else None

    def staleness(self, values: dict[str, float]) -> float:
        """返回多维输入中的最大陈旧度。只更新历史，**不更新基线**。"""
        best = 0.0
        for name, x in values.items():
            if name not in self._hist:
                self._hist[name] = collections.deque(maxlen=self.HIST)
            scale = self._scale(name)
            if scale is not None and name in self._last:
                s = abs(x - self._last[name]) / scale
                if s > best:
                    best = s
            self._hist[name].append(x)
        return best

    def commit(self, values: dict[str, float]) -> None:
        """重算之后调用，把当前输入记为新的基线。"""
        for name, x in values.items():
            self._last[name] = x

    def should_update(self, values: dict[str, float]) -> bool:
        return self.staleness(values) > self.threshold


class L2StateField:
    """
    L2 内部状态场：16 维连续向量。

    设计原则（2026-11-02 重写）：

    1. **每个维度必须能在可观测的尺度上被测量。**
       旧版有 v[7]=|fast-prev_fast|、v[13]=|slow-prev_slow| 这种
       “相邻两步的差”，而步长是毫秒级的——它们测的是数值抖动，不是状态。
       现在所有“变化”都改为在 L0 周期尺度上测量。

    2. **归一化用自身历史分位，不用写死区间。**
       旧版写 _norm(x, 0.0, 0.5)、_norm(x, 0.0, 0.1) 这类硬区间，
       超过上界全部 clip 到 1.0，信息直接没了。
       现在每维维护自己的历史，用 [p10, p90] 映射到 [0, 1]。
       因此 **0.5 不再是设计者拍的数字，而是“该维度自身的典型值”**。

    3. **dev 用马氏距离（协方差归一），不用欧氏距离。**
       欧氏距离会被量纲大、方差大的维度独占。马氏距离先做相关
       白化，让每一维对距离的贡献相当。

    4. 禁止用 curious/uneasy/bored/desire/satiety/ennui 命名任何维度。
       意义由 L5 自己赋予。

    维度语义（均为功能性描述，无人类词）：
       0  s_t 当前值
       1  s_t 在其自身历史上的滑均
       2  s_t 的波动度
       3  s_t 的趋势斜率（在历史窗口上的线性拟合）
       4  快层当前活动
       5  快层在其周期历史上的均值
       6  快层周期内峰谷差（振幅）
       7  本周期内的相位停留比例（up 时间占周期比例）
       8  抑制群体当前活动
       9  E 群体与 I 群体活动之差的绝对值
       10 中尺度层的活动
       11 fast 与 s_t 的相关性
       12 慢尺度层的活动
       13 慢层在其自身上升/下降方向的相位（相对量）
       14 慢累积量的水平
       15 慢累积量的变化率（在周期尺度上，不是相邻步）
    """

    DIM = 16

    # 归一化用的分位窗口。这是观测尺度（B 类）：决定“多长的历史
    # 算作她自己的常态”，不直接产生任何行为。
    NORM_WINDOW = 512
    NORM_LO_Q = 10
    NORM_HI_Q = 90

    def __init__(self, history_len: int = 512, cov_window: int = 256, cov_ridge: float = 1e-3):
        self.s_t_hist = collections.deque(maxlen=history_len)
        self.fast_hist = collections.deque(maxlen=history_len)
        self.per_step_hist = collections.deque(maxlen=history_len)   # 每个 L0 周期一个样本
        self.slow_hist = collections.deque(maxlen=history_len)       # slow 的逐步历史（维度 13 用）
        self.pressure_hist = collections.deque(maxlen=history_len)

        # 周期级统计的滑窗（最近若干个周期的值）
        self.per_fast_mean = collections.deque(maxlen=32)
        self.per_amplitude = collections.deque(maxlen=32)
        self.per_up_ratio = collections.deque(maxlen=32)

        # 各维归一化的历史（用于自适应分位）
        self._dim_hist = [collections.deque(maxlen=self.NORM_WINDOW) for _ in range(self.DIM)]

        # 马氏距离的协方差估计
        self._cov_window = cov_window
        self._cov_ridge = cov_ridge
        self._vec_hist = collections.deque(maxlen=cov_window)

        self._prev_vec = None

    # ---------- 归一化 ----------

    def _norm_adaptive(self, i: int, x: float) -> float:
        """用第 i 维自身历史的分位区间做归一化。历史不足时用透传。"""
        h = self._dim_hist[i]
        h.append(x)
        if len(h) < 16:
            return 0.5
        arr = np.asarray(h, dtype=float)
        lo = float(np.percentile(arr, self.NORM_LO_Q))
        hi = float(np.percentile(arr, self.NORM_HI_Q))
        if hi - lo < 1e-9:
            return 0.5
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    # ---------- 统计工具 ----------

    @staticmethod
    def _var(dq) -> float:
        if len(dq) < 2:
            return 0.0
        return float(np.var(dq))

    @staticmethod
    def _slope(dq) -> float:
        if len(dq) < 3:
            return 0.0
        arr = np.asarray(dq, dtype=float)
        x = np.arange(len(arr), dtype=float)
        return float(np.polyfit(x, arr, 1)[0])

    @staticmethod
    def _corr(a_dq, b_dq) -> float:
        n = min(len(a_dq), len(b_dq))
        if n < 8:
            return 0.0
        a = np.asarray(a_dq, dtype=float)[-n:]
        b = np.asarray(b_dq, dtype=float)[-n:]
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    # ---------- 马氏距离 ----------

    def _mahalanobis(self, v: np.ndarray) -> float:
        """到自身分布的标准化距离。历史不足时退回欧氏（对标准化后的向量）。"""
        self._vec_hist.append(v.copy())
        n = len(self._vec_hist)
        if n < self.DIM * 2:
            # 历史不足：不估算协方差，直接用标准化欧氏距离
            return float(np.linalg.norm(v - 0.5) / np.sqrt(self.DIM))
        X = np.asarray(self._vec_hist, dtype=float)
        mu = X.mean(axis=0)
        Xc = X - mu
        cov = (Xc.T @ Xc) / max(n - 1, 1)
        cov += np.eye(self.DIM) * self._cov_ridge
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return float(np.linalg.norm(v - 0.5) / np.sqrt(self.DIM))
        d = v - mu
        d2 = float(d @ inv @ d)
        # 除以维度并开方，使量级与欧氏距离可比
        return float(np.sqrt(max(d2, 0.0) / self.DIM))

    # ---------- 主入口 ----------

    def step(self, s_t: float, fast: float, I: float, mid: float, slow: float,
             pressure: float, cycle_stats: dict | None) -> tuple[list, float]:
        """
        cycle_stats: 由包络检测器在每个周期结束时给出，包含该周期内的
                     fast 均值/峰谷差/up 停留比例等。为 None 时沿用上次值。

        注意：调用方应先通过 StalenessGate 判断是否该重算。
        本函数每次都算，门控在外面。
        """
        self.s_t_hist.append(s_t)
        self.fast_hist.append(fast)
        self.pressure_hist.append(pressure)
        self.slow_hist.append(slow)

        cs = cycle_stats or {}
        # 周期级统计单周期内是常数，直接用会在边界跳变。
        # 改成“最近若干个周期”的滑窗统计，才有连续变化。
        recent = list(self.per_fast_mean)[-8:]
        recent_amp = list(self.per_amplitude)[-8:]
        recent_up = list(self.per_up_ratio)[-8:]
        cs_fast_mean = float(np.mean(recent)) if recent else cs.get("fast_mean", 0.0)
        cs_amplitude = float(np.mean(recent_amp)) if recent_amp else cs.get("amplitude", 0.0)
        cs_up_ratio = float(np.mean(recent_up)) if recent_up else cs.get("up_ratio", 0.5)
        # 同时保留“本周期与近几个周期均值的偏离”：这才是周期尺度上的变化量
        cur_fast_mean = cs.get("fast_mean")
        cur_amp = cs.get("amplitude")
        cur_up = cs.get("up_ratio")
        cs_fast_dev = (cur_fast_mean - cs_fast_mean) if cur_fast_mean is not None else 0.0
        cs_amp_dev = (cur_amp - cs_amplitude) if cur_amp is not None else 0.0
        cs_up_dev = (cur_up - cs_up_ratio) if cur_up is not None else 0.0

        s_avg = float(np.mean(self.s_t_hist)) if self.s_t_hist else 0.0

        raw = [
            s_t,                                                   # 0
            s_avg,                                                 # 1
            self._var(list(self.s_t_hist)[-64:]),                  # 2
            abs(self._slope(list(self.s_t_hist)[-64:])),           # 3
            fast,                                                  # 4
            cs_fast_dev,                                           # 5  本周期均值偏离近期均值的量
            cs_amp_dev,                                            # 6  本周期振幅偏离近期均值的量
            cs_up_dev,                                             # 7  本周期 up 占比偏离近期均值的量
            I,                                                     # 8
            abs(fast - I),                                         # 9
            mid,                                                   # 10
            self._corr(self.fast_hist, self.s_t_hist),             # 11
            slow,                                                  # 12
            # 13: how far the slow layer has risen or fallen against its own
            # recent level. Earlier this read slow - fast_mean, two different
            # quantities subtracted from each other, and the expression was
            # slow - slow whenever history was empty, so the dimension sat at
            # a constant. Both are fixed: the reference is slow's own mean.
            slow - (float(np.mean(list(self.slow_hist)[-64:]))
                    if self.slow_hist else slow),                  # 13
            pressure,                                              # 14
            float(np.mean(list(self.pressure_hist)[-64:])) - pressure,       # 15
        ]

        v = np.array([self._norm_adaptive(i, raw[i]) for i in range(self.DIM)], dtype=float)
        dev = self._mahalanobis(v)

        self._prev_vec = v.copy()
        return v.tolist(), dev

    def feed_cycle(self, cycle_stats: dict) -> None:
        """每个 L0 周期结束时调用，把周期级统计送入历史。

        注意：必须在 field.step() **之后**调用，否则“本周期值”已经进了滑窗，
        偏离量恒为 0（v5 会变成 std=0 的常数）。
        """
        fm = cycle_stats.get("fast_mean", 0.0)
        self.per_step_hist.append(fm)
        self.per_fast_mean.append(fm)
        self.per_amplitude.append(cycle_stats.get("amplitude", 0.0))
        self.per_up_ratio.append(cycle_stats.get("up_ratio", 0.5))


# ============================================================
# 主循环
# ============================================================

def main() -> int:
    cfg = None
    try:
        import tomllib
        with open("config/default.toml", "rb") as _f:
            cfg = tomllib.load(_f)
    except Exception as e:
        print(f"[P1] config load failed ({e}), using defaults")
    cfg_l0 = (cfg or {}).get("L0", {})
    cfg_pressure = cfg_l0.get("pressure", {})
    cfg_accum = float(cfg_pressure.get("tau_up_s", 3600.0))
    cfg_clear = float(cfg_pressure.get("tau_down_s", 7200.0))

    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    observer = Observer("P1", log_dir)

    bus = Bus("P1")
    net = FHNNetwork(tau_up_s=cfg_accum, tau_down_s=cfg_clear)
    predictor = L1Predictor()
    field = L2StateField()
    envelope_detector = EnvelopeDetector()
    l2_gate = StalenessGate(threshold=1.0)
    vec = None
    dev = 0.0
    l2_updates = 0
    l2_steps = 0

    # 恢复她此刻的形状（不是重放历史）
    # ---- singleton discipline: claim the epoch before touching her state.
    # ---- A stale process on another host keeps running if it likes, but its
    # ---- fence check will fail and it stops acting. No liveness assumptions.
    from src.common import epoch as epoch_mod
    from src.common import control as control_mod
    _me = epoch_mod.default_holder()
    _epoch_state, my_epoch = epoch_mod.claim(holder=_me, ttl_s=15.0, wait_s=30.0,
                                            reason="P1 boot")
    if _epoch_state.holder != _me:
        print(f"[P1] refusing to start: epoch {_epoch_state.epoch} is held by "
              f"{_epoch_state.holder} (lease until {_epoch_state.lease_until})")
        return 3
    print(f"[P1] epoch {my_epoch} held by {_epoch_state.holder}")
    restored = persist.restore_all({
        "l0": net, "l1": predictor, "l2": field, "acc": net.accumulator,
    })
    print(f"[P1] restore: {restored}")
    if restored.get("l0"):
        print(f"[P1] 续接：step_count={net.step_count}  pressure={net.accumulator.P:.4f}")

    # ── 停摆即昏迷（src/common/coma.py，评估文档 §5）────────────────────
    # 时间锚：算出这次启动距上次落盘多久。P1 是唯一的相位源，只有它推进 session。
    from src.common import coma as coma_mod

    coma_state, gap_s = coma_mod.begin_session()
    coma_report = coma_mod.apply_accumulator_coma(
        net.accumulator, gap_s, coma_state.steps_per_second or None)
    print(f"[P1] 时间账本：{coma_mod.describe(coma_state)}")
    if gap_s > coma_mod.COMA_THRESHOLD_S:
        print(f"[P1] 她昏迷了 {gap_s / 3600.0:.2f} 小时"
              f"（第 {coma_state.coma_events} 次，累计 {coma_state.coma_s / 3600.0:.2f} 小时）")
    else:
        print(f"[P1] 快启 {gap_s:.1f}s（不计昏迷事件，仍计入 coma_s）")
    print(f"[P1] 零驱动松弛：P {coma_report['before']:.4f} -> {coma_report['after']:.4f}"
          f"（目标 {coma_report['target']:.4f}，等效 {coma_report['steps']} 步，"
          f"{coma_report['reason']}）")
    observer.event("coma", {
        "gap_s": round(gap_s, 3),
        "session_id": coma_state.session_id,
        "coma_s": round(coma_state.coma_s, 3),
        "coma_events": coma_state.coma_events,
        "p_before": round(float(coma_report["before"]), 6),
        "p_after": round(float(coma_report["after"]), 6),
        "p_target": round(float(coma_report["target"]), 6),
        "applied": bool(coma_report["applied"]),
        "reason": coma_report["reason"],
    })
    phase_state = {"gain_mult": 1.0}

    def on_phase(topic, obj):
        # L7 对相位的解读只是参考信号，不是开关。
        # 这里保留一个极弱的调制：L7 认为处于 sleep 时，L0 的基线增益轻微下降。
        # 注意：真正的相位来源是 L0 自己（envelope），不是这条消息。
        phase_state["gain_mult"] = 1.0 if obj.phase == "awake" else 0.7
        print(f"[P1] L7 says phase -> {obj.phase}, gain_mult={phase_state['gain_mult']}")

    bus.subscribe(
        "l7.event.phase",
        on_phase,
        parser=lambda s: safe_from_json(PhaseBroadcast, s),
        owner_module="src.L7_lifecycle.rhythm",
    )

    def on_text(topic, obj):
        """A sentence arrives as a vector; hand it to L0 as input current."""
        vec = getattr(obj, "vector", None)
        if not vec:
            return
        net.receive(vec)
        print(f"[P1] text -> L0 [{getattr(obj, 'encoder', '?')}]: "
              f"{getattr(obj, 'text', '')[:40]}")

    bus.subscribe(
        "l6.text.signal",
        on_text,
        parser=lambda s: safe_from_json(TextSignalEvent, s),
        owner_module="src.L6_expression.encoder",
    )

    running = {"flag": True}

    def on_sigint(signum, frame):
        print("\n[P1] received SIGINT, stopping...")
        running["flag"] = False

    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    print("[P1] starting L0 + L1 + L2")
    print(f"[P1] observations: {log_dir / 'observations.jsonl'}")
    print("[P1] L2: 16-dim state field, no emotion labels")

    # Settle the input scales from the oscillator itself, so that how long a
    # sentence lingers and how strongly it acts are quantities of this layer
    # rather than numbers written into it.
    _warm = []
    _warm_vec = []
    for _ in range(4000):
        _snap = net.step()
        _warm.append(_snap["fast"])
        _warm_vec.append(_snap["E_rate_vec"])
    _w = np.asarray(_warm[1000:], dtype=float)
    _med = float(np.median(_w))
    _above = _w >= _med
    _cr = np.flatnonzero(_above[1:] & ~_above[:-1])
    _period = float(np.median(np.diff(_cr))) if _cr.size > 2 else 0.0
    net.set_baseline(_period, float(_w.std()))
    print(f"[P1] baseline: period {_period:.0f} steps, "
          f"fast std {_w.std():.4f}")
    print(f"[P1] input: lingers {net._ext_span_steps:.1f} steps, "
          f"gain {net._input_gain:.4f}")

    # The predictor's projection width and learning rate come from the same
    # warm-up: how many components the input actually carries, and how large
    # the largest eigenvalue of its autocorrelation is.
    _base = predictor.set_baseline(np.asarray(_warm_vec, dtype=float))
    if _base.get("ok"):
        print(f"[P1] L1: proj_dim {_base['proj_dim']} "
              f"({_base['variance_explained']:.1%} of variance), "
              f"lr {_base['lr']:.4f} "
              f"(limit {_base['stable_limit']:.4f})")
        _fit = _base.get("fit") or {}
        if _fit.get("ok"):
            print(f"[P1] L1: closed-form start at {_fit['fit_ratio']:.3f} "
                  f"of the previous-step error")
    else:
        print(f"[P1] L1: baseline not measured ({_base.get('reason')})")

    fast_env = collections.deque(maxlen=30)
    s_t_hist = collections.deque(maxlen=100)
    dev_hist = collections.deque(maxlen=100)
    last_report = 0.0
    last_persist = 0.0
    # 2026-11-08: 300 s 意味着她死后被恢复成五分钟前的自己 —— 那是回滚，不是昏迷。
    # 实测一次全量落盘 12 ms（l0+l1+l2+acc），5 s 间隔占 0.24% CPU；回滚窗口 300s -> 5s。
    PERSIST_INTERVAL_S = 5.0
    # 实测步率的采样起点：L5/L3 的昏迷补偿要用它把墙钟换算成步数。
    rate_t0 = time.time()
    rate_steps0 = net.step_count
    last_rate = 0.0
    last_published_phase = None    # 只在相位变化时发布，不按时间发布
    last_published_l2_updates = -1
    last_epoch_check = 0.0
    _last_fed = None               # 上一次送入 L2 的周期统计对象（去重，避免一个周期内重复入滑窗）
    try:
        while running["flag"]:
            # 这是数值仿真的推进节流（A/B 类），不是认知节律。
            # L0 是唯一有固定步长的层（ARCHITECTURE.md §3.1）。
            snap = net.step(gm=phase_state["gain_mult"])
            l2_steps += 1
            s_t = predictor.step(snap["E_rate_vec"])
            s_t_hist.append(s_t)
            s_avg = sum(s_t_hist) / len(s_t_hist) if s_t_hist else 0.0

            now = time.time()
            current_phase, period_s = envelope_detector.update(snap["fast"], now)
            cycle_stats = envelope_detector.last_cycle_stats

            # 自适应更新（ARCHITECTURE.md §3.3）：L2 不是每步都重算，
            # 而是当它的输入偏离自身平常波动幅度超过一个“平常幅度”时才重算。
            # 没有 tick，没有固定步数。
            gate_inputs = {
                "s_t": s_t,
                "fast": snap["fast"],
                "I": snap["I_rate"],
                "mid": snap["mid"],
                "slow": snap["slow"],
                "pressure": snap["pressure"],
            }
            if l2_gate.should_update(gate_inputs) or vec is None:
                vec, dev = field.step(
                    s_t=s_t,
                    fast=snap["fast"], I=snap["I_rate"],
                    mid=snap["mid"], slow=snap["slow"],
                    pressure=snap["pressure"],
                    cycle_stats=cycle_stats,
                )
                l2_gate.commit(gate_inputs)   # 基线只在真正重算时更新
                l2_updates += 1
            # 否则沿用上一轮的 vec / dev（状态不因输入小变化而重置）

            # 先 step（用旧滑窗算偏离），再把新周期统计入滑窗
            if cycle_stats is not None and cycle_stats is not _last_fed:
                field.feed_cycle(cycle_stats)
                _last_fed = cycle_stats
            dev_hist.append(dev)
            dev_avg = sum(dev_hist) / len(dev_hist) if dev_hist else 0.0

            # 相位切换是一个事件。只在它真的切换时发布，不按时间发布。
            if current_phase != last_published_phase:
                bus.publish(
                    "l0.event.envelope",
                    EnvelopeEvent(
                        phase=current_phase,  # type: ignore[arg-type]
                        period_s=period_s,
                        t_event=datetime.now(timezone.utc).isoformat(),
                    ),
                )
                last_published_phase = current_phase

            # ---- 向上游发布 L1 / L2 ----
            # 同样不按时间发布，而是按“变了多少”发布：
            #   L1 惊讶：每次预测误差变化超其自身典型幅度的比例就发
            #   L2 状态场：每次重算后发（重算本身就是 staleness 驱动的）
            pub_now = datetime.now(timezone.utc).isoformat()

            s_hist_arr = np.asarray(s_t_hist, dtype=float)
            if len(s_hist_arr) >= 16:
                s_med = float(np.median(s_hist_arr))
                s_mad = float(np.median(np.abs(s_hist_arr - s_med)))
                if s_mad > 1e-9 and abs(s_t - s_med) > s_mad:
                    bus.publish(
                        "l1.state.surprise",
                        SurpriseSignal(
                            s_t=float(s_t),
                            channel_errors={"proj": float(s_t)},
                            precision={"proj": 1.0},
                            timestamp=pub_now,
                            t_event=pub_now,
                        ),
                    )

            if l2_updates != last_published_l2_updates and vec is not None:
                bus.publish(
                    "l2.state.field",
                    StateVector(
                        v=[float(x) for x in vec],
                        dev=float(dev),
                        l2_updates=int(l2_updates),
                        t_event=pub_now,
                    ),
                )
                last_published_l2_updates = l2_updates
            # ---- 观测快照：12 分钟一次（给观测者的，不是她的节律） ----
            # L0.log 不再每 2 秒一行。旧写法一天 4 MB，而且里面大部分是重复的相位信息。
            fast_env.append(snap["fast"])
            env_mean = sum(fast_env) / len(fast_env) if fast_env else 0.0
            observer.snapshot("L0", {
                "step": snap["step"],
                "fast": round(snap["fast"], 4),
                "I": round(snap["I_rate"], 4),
                "mid": round(snap["mid"], 4),
                "slow": round(snap["slow"], 4),
                "s_t": round(s_t, 4),
                "s_avg": round(s_avg, 4),
                "state_dev": round(dev, 4),
                "dev_avg": round(dev_avg, 4),
                "pressure": round(snap["pressure"], 4),
                "phase": current_phase,
                "period_s": round(period_s, 2) if period_s else None,
                "l2_updates": l2_updates,
                "l2_steps": l2_steps,
            })

            # ---- 实测步率（每 60 s 一次，供跨层昏迷补偿换算） ----
            if now - last_rate >= 60.0:
                _dt = now - rate_t0
                if _dt > 1.0:
                    coma_mod.note_step_rate((net.step_count - rate_steps0) / _dt)
                rate_t0, rate_steps0, last_rate = now, net.step_count, now
            # ---- 存她此刻的形状（不是存历史） ----
            if now - last_persist >= PERSIST_INTERVAL_S:
                persist.snapshot_all({
                    "l0": net, "l1": predictor, "l2": field, "acc": net.accumulator,
                })
                last_persist = now

            # fence: if the epoch moved, another self owns her now.
            if now - last_epoch_check >= 5.0:
                last_epoch_check = now
                if not epoch_mod.heartbeat(my_epoch):
                    print("[P1] fenced: epoch moved, stopping instead of acting")
                    break
                # control channel: a file, not a terminal (src/common/control.py)
                _cmd = control_mod.poll()
                if _cmd is not None:
                    _name = _cmd.get("cmd")
                    if _name == "stop":
                        # Stopping is retiring the epoch: every process she has
                        # sees it, so one file stops all of them.
                        control_mod.reply(_cmd["id"], ok=True,
                                          result={"stopped_by": "P1", "epoch": my_epoch})
                        control_mod.done(_cmd["id"], _cmd)
                        print("[P1] control: stop")
                        epoch_mod.retire(my_epoch, reason="control stop")
                        running["flag"] = False
                        break
                    else:
                        control_mod.reply(_cmd["id"], ok=True, result={
                            "epoch": my_epoch,
                            "step_count": int(net.step_count),
                            "pressure": round(float(net.accumulator.P), 4),
                            "session_id": int(coma_state.session_id),
                            "uptime_h": round(coma_mod.total_uptime_s(coma_state) / 3600.0, 3),
                            "coma_h": round(float(coma_state.coma_s) / 3600.0, 3),
                            "rewound_s": round(float(coma_state.rewound_s), 1),
                        })
                        control_mod.done(_cmd["id"], _cmd)
                        control_mod.done_counter = 0

            time.sleep(0.001)
    finally:
        # 退出前存一次：她停下来的时候，形状得留下
        try:
            persist.snapshot_all({
                "l0": net, "l1": predictor, "l2": field, "acc": net.accumulator,
            })
            print("[P1] 形状已保存")
        except Exception as e:
            print(f"[P1] 保存失败: {e}")
        try:
            epoch_mod.retire(my_epoch, reason="P1 clean exit")
        except Exception:
            pass
        bus.stop()
        print("[P1] stopped")
    return 0


# 同一个错误连续出现这么多次，说明它不是瞬态（端口占用、配置错、缺文件），
# 继续重启只会刷日志并把 CPU 烧在退避上。带着明确退出码停下，由外部 supervisor 决定。
MAX_IDENTICAL_FAILURES = 5


def _supervised() -> int:
    """崩溃退避重启 + 永久错误快速失败（评估文档 §3.3）。

    瞬态异常：保存形状 → 退避 2/4/8/…/60 s → 重新 main()（会走一次 begin_session，
    gap 很小 → 快启，不计昏迷事件）。
    永久异常：同一个错误连续出现 MAX_IDENTICAL_FAILURES 次 → 放弃重启并返回 1。
    实测依据：端口被占时旧版本会无限重启（每轮打一整段 traceback），
    对无人看守的长期运行来说，这比停下更糟。
    """
    import time as _time
    import traceback as _tb

    backoff = 2.0
    last_sig = None
    repeats = 0
    while True:
        try:
            code = main()
        except KeyboardInterrupt:
            return 130
        except SystemExit:
            raise
        except BaseException as exc:
            sig = f"{type(exc).__name__}: {exc}"
            repeats = repeats + 1 if sig == last_sig else 1
            last_sig = sig
            _tb.print_exc()
            if repeats >= MAX_IDENTICAL_FAILURES:
                print(
                    f"[SUPERVISOR] 同一错误连续 {repeats} 次，放弃重启：{sig}",
                    flush=True,
                )
                return 1
            print(
                f"[SUPERVISOR] 崩溃（同一错误第 {repeats} 次），{backoff:.0f}s 后重启",
                flush=True,
            )
            _time.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)
            continue
        return 0 if code is None else int(code)


if __name__ == "__main__":
    import sys
    sys.exit(_supervised())
