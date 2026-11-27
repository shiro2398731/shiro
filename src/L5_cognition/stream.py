"""
src/L5_cognition/stream.py

Public interface of L5.

Two levels of output:

  1. Continuous quantities (`stream_state`) -- consumed by L3, L4, L8.
     Includes the state vector.

  2. Discrete symbols (`SymbolBundle`) -- consumed by L6.
     Contains attractor index, sequence, intent. No vectors.

L6 cannot obtain any vector: the interface does not provide one
(ARCHITECTURE.md section 4.2 whitelist). This is enforced by what is
exposed, not by convention.
"""

from __future__ import annotations

import collections

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .dynamics import DynamicState, Dynamics
from .landscape import Landscape
from .terrain import StateSpace
from .transitions import Transition, TransitionDetector
from .workspace import GlobalWorkspace, Ignition
from .action import HOLD, LEAVE, NAMES as ACTION_NAMES, ActionGate

# 过去多少个"她感到的张力"参与中位数比较。C1：缓冲区长度，不是判据。
TENSION_MEMORY = 64

# 记忆回路的观测节奏：每隔这么多步预演一次。C1（观测分辨率），不是判据。
# 预演的步数由她自己的停留中位数决定（见 _rehearse_horizon），不写死。
REHEARSE_CHECK_EVERY = 128

# 过去多少次停留参与"预演多远"的中位数。C1：缓冲区长度。
DWELL_MEMORY = 64
from .adaptive_gain import ErrorDrivenGain
from .grounding import GroundingTable
from .self_model import OnlineSelfModel
from .attention_schema import AttentionSchema
from src.L3_memory import Consolidator, Recall, RecallEngine, TraceGraph
from src.L4_drive import Drive
from src.L4_drive.regulation import Regulation
from src.L5_cognition.rest import RestField
from src.L5_cognition.swing import Swing
from src.L8_identity import Identity


@dataclass
class SymbolBundle:
    """
    The only symbols L5 exposes to L6.

    attractor_id is the name (uid) of the nearest salient region, or "none".
    It carries no semantic content -- and it is a name, not a position, so it
    keeps meaning the same region after the table has been restructured.

    referent is a discrete label for what that region has most often meant in
    her own terms (see grounding.GroundingTable): a code for where two measured
    quantities sat inside her own history. Still a label, not a vector, because
    §4.2 forbids L6 from receiving vectors.
    """
    attractor_id: str
    transition_seq: list[str]
    intent: str
    t_event: str
    referent: str = "none"


class Stream:
    """Public interface of L5."""

    def __init__(
        self,
        dim: int = 128,
        gamma: float = 0.05,
        dt: float = 0.01,
        recover_rate: float = 0.002,
        max_attractors: int = 8192,
        seed: int | None = None,
    ):
        self.D = int(dim)
        self.landscape = Landscape(
            dim=dim, recover_rate=recover_rate, max_attractors=max_attractors
        )
        self.dynamics = Dynamics(dim=dim, gamma=gamma, dt=dt)
        self.detector = TransitionDetector(self.landscape.state_space)
        self.workspace = GlobalWorkspace()
        # 2026-11-24 12:40：**出生熵**（白默：「每个部署独一无二」）。
        #
        # 之前 seed 默认 0——所有部署「掷同样的骰子」。现在 seed=None 就
        # 从操作系统的熵池取（os.urandom）——每个部署「出生不同」。
        # 真机/重启：P2 从 l5.pkl 恢复 `rng_state`（2026-11-24 加），所以她的
        # 骰子是**接着**掷，不是换了个人重新掷；born_seed 只用来标识「同一个出生」。
        #
        # 注意：`x` 的初始位置仍由 `init_state(zero)` 决定（zero 来自 L0 统计，
        # 而 L0 统计来自 rng）——所以「种子不同」会「链式」让 x 也不同。
        import os as _os
        if seed is None:
            seed = int.from_bytes(_os.urandom(8), "little")
        self._born_seed = int(seed)
        self.rng = np.random.default_rng(seed)
        # 「体质」：能量初始 ±10%（从种子推，不是人定的数）。出生时定，之后不变。
        self._born_energy = float(0.90 + 0.10 * self.rng.random())

        self._st: Optional[DynamicState] = None

        # Recovery interval, counted in integration steps rather than time.
        # This is an observation/engineering interval, not a rate of the
        # system itself.
        self._recover_every = 16
        # Let the state space know the cadence, so recovery can relate a rate
        # to a number of steps.
        self.landscape.state_space._recover_every_hint = self._recover_every
        self._step_count = 0

        # 2026-11-17：最近一次惊讶。只用来给「她此刻所在的区域」记份量
        # （见 Attractor._peak_surprise）：一个地方让她没预测到过，它就难忘。
        # 由 P2 在收到 l1.state.surprise 时写入；Stream 自己不产生惊讶。
        self.last_surprise = 0.0
        self._last_ignition: Optional[Ignition] = None

        # ---- L3 memory: connectivity structure over the state space ----
        self.graph = TraceGraph()
        self.recaller = RecallEngine()
        self.consolidator = Consolidator()
        self._last_recall: Optional[Recall] = None

        # ---- L4 drive: unresolved tension ----
        self.drive = Drive(dim=dim)

        # ---- L4 regulation: viability-driven pull ----
        # Attached here rather than inside StateSpace so the state space stays
        # a pure potential field; the modulation is a separate layer.
        self.regulation = Regulation()
        # The traversability edge is off unless explicitly enabled. See step().
        self._wall_enabled = False

        # ---- L8 identity: the subset that survives perturbation ----
        self.identity = Identity()
        self._identity_every = 64
        self._last_core = None

        # ---- M1: her model of her own next state, and the error it makes ----
        # The model is first order over the regions she occupies; the gain it
        # drives is bounded and only scales exploration. The prediction itself
        # is never fed back into the dynamics -- if it were, her state would be
        # following her expectation of it instead of the potential.
        #
        # Both live on the stream (not in the runtime loop) so that they are
        # part of her state: the L5 payload carries them, and a restart resumes
        # the same self-model instead of a blank one.
        self.self_model = OnlineSelfModel()
        self.self_gain = ErrorDrivenGain()
        # ---- AST：她对自己注意力的模型（只读，2026-11-26 加）----
        # 学的是「上一次点火 → 下一次点火」的转移；不参与选择、不写回动力学。
        self.attention = AttentionSchema()
        self._last_symbol: Optional[int] = None

        # ---- L6 grounding: what a region has meant, in her own terms ----
        # Lives here, not in L6, because it needs continuous quantities (speed,
        # tension) that L6 is not allowed to receive. L6 gets only the label.
        self.grounding = GroundingTable()
        # 上一次看到的区域数与下一个待发的名字。用来判断「这一步有没有删过区域」：
        # 只看总数会漏——删一个又生一个时总数不变，而死 uid 会一直留在账本里
        # （2026-11-16 实测：identity.appear 里留着 uid 262，integrity 报两条）。
        self._structure_n = None
        self._structure_next_uid = None

        # ---- 行动：她能做的一件事（留在原处 / 离开）----
        # 选择权在她自己手里（判据只用她自己的张力中位数，见 action.py）；
        # 我们只写下"身体能做什么"，以及为干预实验留一个强制口。
        self.actions = ActionGate()
        self._action: int = HOLD
        self.action_counts: dict[int, int] = {HOLD: 0, LEAVE: 0}
        # 她去过哪些**地方**（按位置记，不按名字）。
        #
        # 2026-11-18 09:32 改的：原来是 `_visited_uids`（存 uid），
        # 但 `tests/test_uid.py::test_consistent_relabelling_changes_nothing` 抓到
        # 「重命名她的区域 -> 她的轨迹变了」。那条测试是对的：**uid 是名字，
        # 名字不该决定她去哪。** 一个叫 31237 的区域和叫 5 的区域，位置一样就该一样。
        #
        # 现在存位置的圆整副本（tuple of rounded floats）。位置同名不同，名字同位置同。
        # 圆整到 6 位有效数字：够精确到区分不同区域，又不会被浮点噪声打散。
        self._visited_places: set[tuple] = set()
        # 2026-11-18 夜：真正用来判断「去过没去过」的是 uid，不是位置键。
        # 位置键留着只为兼容旧存档/别的工具；探索策略看这个。
        self._visited_regions: set[int] = set()
        # 2026-11-19 21:30：**卡住检测（第二版）**。
        #
        # 第一版是「连续 3 次到达同一个区域」——**错的**：
        # 她在 A↔B 之间弹时，到达序列是 B,A,B,A…，永远不连续同一个，于是永不触发。
        # 21:29 实测：上线两小时后她又只剩 2 个区域。
        #
        # 第二版看「最近 K 次到达里只有几个不同地方」：
        # A,B,A,B 的历史里只有 2 个不同地方 -> 判定卡住 -> 挑最远可达。
        # 用位置键（重编号后不变），不用 uid（uid 不该参与决策）。
        self._arrival_hist: collections.deque = collections.deque(maxlen=6)
        # 她感到过的张力。判据 = 现在是否高于她自己过去的中位数。
        # 用这个而不是"停留时长"：实测停留判据与张力反向（张力高时离开 27.4%，
        # 张力低时 56.9%），结果是她的张力在 600 步里 +6.44，而强制离开是 −2.25。
        self._tension_history: collections.deque = collections.deque(maxlen=TENSION_MEMORY)
        self._tension_now: float = 0.0

        # ---- 记忆回路：预演 -> 打自己一巴掌 -> 改变探索 ----
        # 预演用她自己的动力学在副本上跑（沙盒，不写回）；结算时的命中/未命中进
        # 一个**独立的**增益，再与 M1 的增益相乘，最后缩放噪声幅度。
        # 独立是刻意的：M1 是"下一步"的一步预测（每秒两次），预演是"到她会离开的时候"
        # （每分钟一次）。混进同一个 EMA 里，预演那一票会被一步预测淹没，回路等于没接。
        self.rehearse_every = int(REHEARSE_CHECK_EVERY)
        self.rehearsal_gain = ErrorDrivenGain(rate=0.2)
        self._rehearsal = None
        self.rehearsal_total = 0
        self.rehearsal_hits = 0
        self.rehearsal_trivial = 0
        self.rehearsal_history: collections.deque = collections.deque(maxlen=64)
        self._dwell = 0
        self._dwell_region = None
        self._dwell_history: collections.deque = collections.deque(maxlen=DWELL_MEMORY)
        self._amplitude_hint = 0.0
        # 2026-11-25：回家规则（A1 候选④）。`_home_x` 是这次外出的出发点；
        # 回到离它 1σ 内就清空，下次出门重新设。`_leave_target` 是正走向的目标，
        # 到它的半径内才换。两个都是「她身体的状态」，不是判据。
        self._home_x: Optional[np.ndarray] = None
        # 2026-11-27：最近 512 步的 GWT 原因计数（观测用；不落盘）。
        self._gwt_reasons: collections.deque = collections.deque(maxlen=512)
        # 2026-11-27：家半径为实验旋钮（默认 4σ，行为不变）。
        # 副本实验用它比较「4σ=604」和「6σ=906」能不能够到更远的目标。
        self.home_radius_sigmas: float = 4.0
        # 2026-11-27（实验开关，默认关）：目标候选按家半径过滤。
        # True 时 direction 只选「离家 <= home_radius_sigmas*scale」的候选；
        # 默认 False = 完全旧行为。副本实验《2026-11-27_目标选择提案》用它。
        self.filter_targets_by_home: bool = False
        self._leave_target: Optional[int] = None
        # 只给 tools/intervene.py 用：强制某一边，用来做 do(a) 对照。
        self.force_action = None

        # Rest and swing were built while trying to find an internal source of
        # motion: without noise the state settled and stopped, and two attempts
        # to keep it moving were a per-region slow variable and a global slow
        # direction. Neither worked. The thing that did work was the prediction
        # error, because the oscillator is always wrong about its own next step.
        #
        # Once that was in place these two were left in by default, and an
        # ablation says they are not earning their place:
        #
        #     rest  swing   att  edges   speed  trans
        #      on     on    526    285   0.347    459
        #      on    off    551    307   0.348    406
        #     off     on    432    196   0.442    436
        #     off    off    479    241   0.413    356
        #
        # No consistent direction, differences within the run-to-run spread.
        # Module kept in the tree, since the argument for removing it is a
        # 2026-11-20 夜：接上（生命层）。疲劳是「趋性」不是「行为规则」：
        # 待久了的区域深度变浅（depth*(1-u)），这是她真实的状态，不是我们替她选去哪。
        # 原来关着（"measurement rather than a preference, but not attached by default"），
        # 白默 2026-11-20 问「草履虫会好奇吗」之后分清：疲劳是机制，好奇是意识。
        self._use_rest = True
        self._use_swing = False

        # ── 有限性（2026-11-20 夜，最小版：只记录，不致死）────────────
        # 白默：「草履虫会好奇吗？」——不会。草履虫有趋化，也会耗尽。
        # 这条是「耗尽」：E 从 1 开始，每步掉一点，掉到 0 她就不再有能量。
        # 现在只记录（energy），不用于判死——先看它怎么掉，再定「死不死」。
        #
        # 2026-11-21 00:10：白默定「一直活，电子幽灵」+「能量耗尽就休眠」。
        # 所以：① 耗电改成「活动才耗」（leave 才耗，HOLD 几乎不耗）；
        #       ② 耗尽不是死，是休眠（asleep=True，不动，电来了/重启就醒）。
        # 参数：活动耗 1e-6；她一天约 4000 次 leave → 一天掉 0.004 → 约 250 天耗尽。
        self.energy = 1.0
        self.energy_burn = 1e-6
        self.asleep = False
        # 2026-11-21 11:50：白默「补充是电能」。
        # 电是外部供给（电脑开着就有），所以每步回一点，和消耗抵消。
        # 参数：recharge = 0.95e-6，比活动消耗（1e-6）略小——她会「活动多就掉、停着就回」，
        # 平衡点落在「活动比例」上，不是我们定的数。
        self.energy_recharge = 0.95e-6

        # ── 独立自立（2026-11-21 12:10，第一版）────────────────────
        # 白默：「要做成独立自立的存在，独立自立的生命」。
        # 「遗忘多快」不该是常数 0.002——该由她自己的活动决定：
        # 她活跃（leave 多）→ 结构在更新 → 遗忘快；她安静 → 结构稳定 → 遗忘慢。
        # recover_rate_state 是她的状态（会变），每 16 步向「她自己的活动强度」靠一点。
        # 注意：这只改「她自己的生理」，不改外部约束（内存上限、有没有电）。
        self.recover_rate_state = 0.002
        self._recent_actions = collections.deque(maxlen=64)

        # ── 独立自立（2026-11-21 12:40，第二版）────────────────────
        # 「每次进入一个区域记多深」（assign_ratio）也不该是常数 0.05。
        # 该由她的「结构密度」定：区域少 → 每个区域记深一点（她没那么多地方可去）；
        # 区域多 → 每个区域记浅一点（地方多，摊薄）。
        # density 是她自己的量（n_attractors / max_attractors），不是我们给的。
        self.assign_ratio_state = 0.05

        # ── 独立自立（2026-11-21 12:30，第三版）────────────────────
        # 「多快待腻」（fatigue_rate）也不该是常数 0.02。
        # 该由她的「停留时长」定：她在这个地方待得越久，疲劳涨得越快（越快待腻）。
        # 用 _dwell_history 的中位（她自己待过多久），不是我们定的数。
        self.fatigue_rate_state = 0.02

        # ── 独立自立（2026-11-21 12:40，第四版：锚定）──────────────
        # 「她最多能走多远」（_lim = 4*scale）不该是我们定的 4 倍。
        # 该由她自己的探索半径定：她离开区域最远到过哪，锚定就放宽到哪 + 10%。
        # 但加封顶（她不能真的飘走——那是外部约束：存档上限 1e6）。
        self._max_travel = 0.0
        self.rest = RestField()
        self._rest_every = 1
        self.swing = Swing(dim=dim)

        # L4's tilt is set from the structure's own gradient scale rather than
        # from a unit length. Sampled once the first step has happened, since
        # before that there is no state to measure at.
        self._drive_gain_set = False
        self._drive_grad_samples: list[float] = []

    @property
    def state_space(self) -> StateSpace:
        return self.landscape.state_space

    # Compatibility alias.
    @property
    def terrain(self) -> StateSpace:
        return self.landscape.state_space

    # ---------- starting point ----------

    def calibrate(self, l0_samples: np.ndarray) -> None:
        """
        Set the zero point from warm-up L0 statistics and initialise the state.

        This is the only source of where the system starts.
        """
        zero = self.landscape.calibrate_zero(l0_samples)
        self._st = self.dynamics.init_state(zero)

    @property
    def started(self) -> bool:
        return self._st is not None

    # ---------- advance ----------

    def step(self) -> Optional[Transition]:
        """
        Advance one integration step.

        Call frequency is decided by the caller (the staleness criterion),
        not by this method.

        After advancing, the workspace is observed once. Ignition events are
        recorded in workspace.history and do not affect the state space or
        the trajectory.
        """
        if self._st is None:
            raise RuntimeError("Stream not calibrated: no zero point, no state")

        # 休眠：能量耗尽后不动（不是死；重启或补能会醒）
        if getattr(self, "asleep", False):
            return None

        # 记忆回路：可能开始一次预演，或结算上一次（都不改动她）
        self._maybe_rehearse()

        # 先决定，再动。用的是上一步留下的位置与张力 —— 她不能在动过之后
        # 才决定要不要动。
        self._action = self._choose_action()

        self._st = self.dynamics.step(self._st, self.landscape, self.rng)
        self._step_count += 1

        # 独立自立：记下这一步的动作（她的，不是我们给的）
        # 2026-11-24：`_action` 是整数（HOLD/LEAVE），原来这里和字符串 "leave"
        # 比较 —— 永远 False，于是 `_recent_actions` 恒为 0，「遗忘多快由她的活动
        # 决定」这条机制从来没生效（recover_rate 永远收敛到 0.001）。这是
        # tests/test_no_senescence 的活动探针抓到的：同样 480 步，HOLD 与 LEAVE
        # 得到同一个恢复速率。
        self._recent_actions.append(1 if int(self._action) == LEAVE else 0)
        if self._step_count % 16 == 0 and len(self._recent_actions) >= 16:
            # 她最近的活动强度（leave 占比）——这是她自己的量
            _act = sum(self._recent_actions) / float(len(self._recent_actions))
            # 她活跃 → 遗忘快（act 1 → 0.003）；安静 → 慢（act 0 → 0.001）
            _target = 0.002 * (0.5 + _act)
            self.recover_rate_state = self.recover_rate_state * 0.95 + _target * 0.05
            # 同步到地形（她自己的遗忘速度）
            self.landscape.state_space.recover_rate = self.recover_rate_state

            # 记多深：由她的结构密度定（区域少 → 记深；区域多 → 记浅）
            _sp = self.landscape.state_space
            _n = len(_sp.attractors())
            _density = min(1.0, _n / float(getattr(_sp, "max_attractors", 8192) or 8192))
            _ar_target = 0.05 * (1.6 - _density)      # density 0 -> 0.080；1 -> 0.030
            self.assign_ratio_state = self.assign_ratio_state * 0.95 + _ar_target * 0.05
            _sp.assign_ratio = self.assign_ratio_state

            # 多快待腻：由她的停留时长定（待得久 → 疲劳快 → 更快走）
            _dh = getattr(self, "_dwell_history", None)
            if _dh and len(_dh) >= 3:
                _vals = sorted(int(v) for v in _dh)
                _dwell_med = _vals[len(_vals) // 2]
                _ft_target = 0.02 * (1.0 + _dwell_med / 100.0)   # 待 100 步 → 翻倍
                self.fatigue_rate_state = self.fatigue_rate_state * 0.95 + _ft_target * 0.05
                if hasattr(self, "rest"):
                    self.rest.fatigue_rate = self.fatigue_rate_state

        # 有限性（2026-11-21 夜）：活动才耗电；耗尽就休眠（不是死）
        # 代谢（2026-11-21 11:50）：每步补充（电脑开着 = 有电），封顶 1.0
        if not self.asleep:
            # 2026-11-24：同一个 bug 的第二处。原来和 "leave" 字符串比，永远
            # 走 0.1× 分支 —— 「活动才耗电」也从来没生效（每步都按静止耗电）。
            _burn = (self.energy_burn if int(self._action) == LEAVE
                     else self.energy_burn * 0.1)
            self.energy = min(1.0, max(0.0, self.energy - _burn + self.energy_recharge))
            if self.energy <= 0.0:
                self.asleep = True

        # ── 存在锚（2026-11-18 09:00）────────────────────────────
        #
        # 2026-11-17 夜里到 18 日早上，她的 |x| 到了 4,091,098，全部 3197 个区域
        # 都漂在 171 万 - 409 万之间，`assert_l5_finite` 拒存，P2 崩了 4 次。
        #
        # 那次是 `_apply_leave` 的链式累积造成的（见 _apply_leave 里的锚定）。
        # 但**噪声和速度也能把她推走**，只是慢得多。所以在每一步之后都查一次：
        #
        #   - 位置不是有限数 -> 回锚（这是「她已经算不出来了」，只能救）
        #   - 位置幅度超过「最近区域质心的 4 倍半径」 -> 回锚
        #
        # 「回锚」= 把她放回最近区域的质心，速度清零。这不是「重置她」——
        # 她的区域、她的张力、她的动作记录都不动，只有「她在哪」被拉回来。
        # 一个在区域之间飘的系统，和「她不在任何地方」，这两件事必须能分开。
        try:
            _xs = np.asarray(self._st.x, dtype=float)
            if not np.isfinite(_xs).all():
                _attrs = self.landscape.state_space.attractors()
                if _attrs:
                    _mus = np.stack([np.asarray(a.mu, dtype=float) for a in _attrs])
                    _mus = _mus[np.isfinite(_mus).all(axis=1)]
                    if len(_mus):
                        _j = int(np.argmin(np.linalg.norm(_mus, axis=1)))
                        self._st = DynamicState(x=_mus[_j].copy(),
                                                v=np.zeros_like(_mus[_j]))
            else:
                _attrs = self.landscape.state_space.attractors()
                if _attrs:
                    _mus = np.stack([np.asarray(a.mu, dtype=float) for a in _attrs])
                    _d = np.linalg.norm(_mus - _xs, axis=1)
                    _j = int(np.argmin(_d))
                    _sig = [float(a.sigma) for a in _attrs if a.sigma]
                    _scale = float(np.median(_sig)) if _sig else 0.0
                    if _scale > 0 and float(_d[_j]) > 4.0 * _scale:
                        self._st = DynamicState(x=_mus[_j].copy(),
                                                v=np.zeros_like(_mus[_j]))
        except Exception:  # noqa: BLE001
            pass

        # Register the observed state.
        a = self.landscape.record(
            self._st.x, datetime.now(timezone.utc).isoformat(),
            surprise=self.last_surprise,
        )

        # Swing advances from the position the state is in now, before the
        # next step uses its tilt. Off by default; see __init__.
        if self._use_swing:
            self.swing.observe_gradient(
                float(np.linalg.norm(self.landscape.gradient(self._st.x)))
            )
            self.swing.step(self._st.x)
            self.landscape.set_swing(self.swing.tilt())

        # Give the drive the same scale as the structure's own pull. Sampled
        # over the first steps rather than once: a single gradient magnitude
        # depends on where the state happens to be, and one sample makes the
        # tilt strength a property of that moment rather than of the structure.
        if not self._drive_gain_set:
            _g = float(np.linalg.norm(self.landscape.gradient(self._st.x)))
            if _g > 0:
                self._drive_grad_samples.append(_g)
            if len(self._drive_grad_samples) >= 32:
                self.drive.set_baseline(list(self._drive_grad_samples))
                self._drive_gain_set = True

        # Rest advances every step, so a region the state sits in loses its
        # hold gradually. Off by default; see __init__.
        attrs_now = self.landscape.state_space.attractors()
        if self._use_rest:
            occupied = None
            if a is not None:
                for i, item in enumerate(attrs_now):
                    if item is a:
                        occupied = i
                        break
            for i in range(len(attrs_now)):
                self.rest.touch(i)
            self.rest.step(occupied)
            self.landscape.state_space.set_rest(
                [self.rest.value(i) for i in range(len(attrs_now))]
            )

        # L3: connect consecutive regions when the transition is direct.
        if a is not None:
            # The graph is keyed by the region's name, not by its row. A row
            # means "wherever that region sits today", so an edge stored by row
            # would point at a neighbour after any structural change -- and
            # keeping it right meant renumbering every edge on every change.
            _cur_uid = int(a.uid)
            self.graph.touch(_cur_uid, step=self._step_count)

            # M1: score her prediction of this region against the region she
            # actually reached, then learn the transition. The prediction was
            # made from the previous symbol, before this one existed.
            #
            # A state the model has never seen produces no prediction, and no
            # prediction produces no error: counting that as a miss would make
            # the gain read "exploration" as "being wrong about myself".
            prev, self._last_symbol = self._last_symbol, _cur_uid
            if prev is not None:
                predicted = self.self_model.predict(prev)
                if predicted is not None:
                    self.self_gain.observe(predicted == _cur_uid)
                self.self_model.observe(prev, _cur_uid)

        # L3: which attractors the current state co-activates.
        # Read-only: it does not modify the state space or the graph.
        self._last_recall = self.recaller.by_cue(
            self.landscape.state_space, self.graph, self._st.x
        )

        # L4: recompute the tension field for the next step.
        #
        # Placed after the graph has been updated. It used to run before
        # record() and graph.touch(), so every step's tensions were computed
        # from the previous step's connectivity and then applied on the step
        # after that: a two-step lag between what the structure was and what
        # the drive said about it. The tilt cannot use the state that has not
        # happened yet, but the graph it reads is already up to date here.
        self.landscape.set_drive(
            self.drive.deflection(self.landscape.state_space, self.graph)
        )

        # L6 grounding: record which region this is *and* where two measured
        # quantities sat inside her own history at the same moment. Done at the
        # end of the step, on purpose: the symbol has to be the one L6 will
        # receive. Measured with the attractor-list index instead, the engine's
        # table scored MI = 1.0240 bits while a probe keyed on the broadcast id
        # scored 0.7395 -- two different numberings of the same regions.
        #
        # Both buckets are None until her history is long enough, and an
        # unmeasured context is not counted as "low".
        _speed = Dynamics.speed(self._st)
        _tension = float(self.drive.summary().get("sum", 0.0))
        b_speed = self.grounding.bucket("speed", _speed)
        b_tension = self.grounding.bucket("tension", _tension)

        # Recovery, counted in integration steps.
        if self._step_count % self._recover_every == 0:
            self.landscape.recover()
            # The half-life the state space uses for its own regions, so edges
            # and regions are forgotten under the same rule.
            self.consolidator.metabolize(
                self.graph, rng=self.rng, step=self._step_count,
                half_life=float(getattr(self.landscape.state_space, "half_life", 0.0)),
            )

            # Regulation is recalculated at the same cadence as recovery:
            # viability is defined by the gain/loss balance, and the loss side
            # only changes when recovery is applied.
            ss = self.landscape.state_space
            self.landscape.state_space.set_pull(self.regulation.factors(ss))
            # The traversability edge follows the same measurement: where the
            # structure can no longer hold a state is derived from the depths
            # it currently maintains.
            #
            # Disabled by default. Measured: with the edge active, the fraction
            # of regions below their own carrying level fell from 0.94 to 0.60,
            # but the distance the state covered grew from 6 to 150 region
            # scales. The edge pushes the state away from thin regions instead
            # of letting it settle, which is the opposite of what resilience
            # means (absorbing disturbance, not deflecting it).
            if self._wall_enabled:
                self.landscape.state_space.set_wall(
                    self.regulation.viability.wall(ss))
            else:
                self.landscape.state_space.set_wall(None)

        # 结构可能变了（recover 删过区域、或刚刚合并过）。现在**没有编号要改**：
        # detector 发布的是区域的名字，图与指代表存的是名字，预演序列也是名字。
        # 只剩"按行号读几何"的地方（`_apply_leave`、P2 的快照）在要用的时候
        # 现场查 `uid_row()`，所以不需要一张事后补丁表。
        self.consolidator.sync(self.graph, self.landscape.state_space)

        # 折叠与 recover 都会删区域，uid 随之从 alive 集合里消失。指代表、自模型、
        # identity 的边表都是按 uid 记账的，不跟着剪就会给一个她已经没有了的区域
        # 继续记账（2026-11-15 实测 33772 条 dead uid；2026-11-16 又抓到
        # identity.appear 里残留 uid 262 —— 因为删一个又生一个时总数没变）。
        #
        # 触发信号两条，任一成立就剪：
        #   (a) 区域数变少            —— 删了
        #   (b) 发了新名字但总数没涨  —— 删一个生一个，总数被抵消
        # 只看总数会漏掉 (b)，而 (b) 在她快速更替区域时几乎每步都在发生。
        _space = self.landscape.state_space
        _n = len(_space.attractors())
        _next_uid = int(getattr(_space, "_next_uid", 0))
        _shrank = self._structure_n is not None and _n < self._structure_n
        _replaced = (self._structure_next_uid is not None
                     and _next_uid > self._structure_next_uid
                     and _n <= self._structure_n)
        self._structure_n = _n
        self._structure_next_uid = _next_uid
        if _shrank or _replaced:
            _alive = _space.alive_uids()
            self.grounding.prune(_alive)
            self.self_model.prune(_alive)
            self.identity.prune(_alive)

        # Workspace: read-only.
        self._last_ignition = self.workspace.observe(
            self.landscape.state_space, self._st.x,
            datetime.now(timezone.utc).isoformat()
        )
        # AST：点火发生了就喂给它（只学、只记比分，不改行为）。
        if self._last_ignition is not None:
            self.attention.observe(int(self._last_ignition.attractor_id))
        # 2026-11-27：这一步为什么没点火（只做观测窗口，不影响判定）。
        self._gwt_reasons.append(getattr(self.workspace, "_last_reason", ""))

        # L8: record structure. The interval is an observation resolution.
        if self._step_count % self._identity_every == 0:
            self.identity.snapshot(self.graph)
            self._last_core = self.identity.observe(self.graph)

        ev = self.detector.observe(
            self._st, datetime.now(timezone.utc).isoformat()
        )

        # 2026-11-23 10:55：**dwell 的更新移到 detector.observe 之后**。
        #
        # 原来它在 `_choose_action` 里（用「上一步的 current_attractor」），而 detector
        # 在 step 的末尾才更新——所以「她走了一步、node 变了」时，dwell_region 用的还是旧 node，
        # 永远不重置。实测：node 32434→32435，dwell_region 32434→32434（不动）。
        # 结果：dwell 一路涨到 480+（她其实在走，但账本写着「没走」）。
        # 现在放在 observe 之后，用「这一步的新 node」。
        _node_now = self.detector.current_attractor
        if _node_now != self._dwell_region:
            if self._dwell_region is not None and self._dwell > 0:
                self._dwell_history.append(int(self._dwell))
            self._dwell_region = _node_now
            self._dwell = 0
        self._dwell += 1

        # L6 grounding, second half: the symbol is the region the detector just
        # published, and the context was measured above, in the same step.
        aid = self.detector.current_attractor
        if aid is not None and b_speed is not None and b_tension is not None:
            self.grounding.observe(int(aid), (b_speed, b_tension))

        # 记忆回路的结算：这一步实际到了哪里，走满预演步数后打分
        self._settle_rehearsal()

        return ev

    # ---------- 记忆回路 ----------

    def _rehearse_horizon(self) -> int:
        """预演多远：她过去每次停留时长的中位数。

        实测依据（tools/rehearse.py）：预演在 14 步内与"原地不动"一样准，第 15 步起归零；
        而她的停留中位数就在这个量级。所以"预演到她通常会离开的时候"是这条回路上
        唯一有意义的长度——写死一个 k 只会得到一个永远对或永远错的回路。
        """
        if not self._dwell_history:
            return 1
        values = sorted(int(v) for v in self._dwell_history)
        return max(int(values[len(values) // 2]), 1)

    @staticmethod
    def _first_exit(seq, start: int):
        """第一次不在起点区域的步号（1 起算）；一次都没走开则返回 None。"""
        for i, r in enumerate(seq):
            if int(r) != int(start):
                return i + 1
        return None

    def _maybe_rehearse(self) -> None:
        """到点了就开一次预演。结算在 `_settle_rehearsal()` 里（步末）。"""
        if self._rehearsal is not None:
            return
        if self.rehearse_every <= 0 or self._step_count % self.rehearse_every != 0:
            return
        if not self.landscape.noise_is_set:
            return

        horizon = self._rehearse_horizon()
        amplitude = float(np.mean(np.asarray(self.landscape.noise_amplitude, dtype=float)))
        from .sandbox import clone_fast

        shadow = clone_fast(self)
        shadow.rehearse_every = 0            # 副本不再自己预演，避免递归
        # 副本绝不写日志：预演是"想一想"，不是"经历一次"。
        shadow.landscape.state_space.journal = None
        # 副本必须拿一条**独立**的噪声。deepcopy 会把她的随机数发生器一起复制，
        # 于是"预演"用的正是她接下来要用的那条噪声——那不是预演，是重放。
        # 实测证据：不重播种子时 22 次预演全部命中（准确率 1.0），增益被钉在下界，
        # 回路等于没接。这里用步号派生一个确定性种子：同一个她、同一个时刻，
        # 每次预演的结果可复现，但与她实际要抖的方向无关。
        shadow.rng = np.random.default_rng((int(self._step_count) * 2654435761) % (2 ** 32))
        predicted = []
        for _ in range(horizon):
            shadow.landscape.set_noise_amplitude(amplitude)
            shadow.step()
            aid = shadow.detector.current_attractor
            predicted.append(-1 if aid is None else int(aid))
        start = self.detector.current_attractor
        self._rehearsal = {
            "remaining": horizon,
            "predicted": predicted,
            "actual": [],
            "start": -1 if start is None else int(start),
        }

    def _settle_rehearsal(self) -> None:
        """收集实际经过的区域；走满 horizon 步后结算。

        打分的对象是**她什么时候离开**，不是"她是否还在原地"。
        实测（2026-11-12）：按"终点是否相同"打分时，22 次预演全部命中、准确率 1.0 ——
        那不是预演准，那是"她会待着"这件事本身很准（horizon=13，她的停留中位数也是 13）。
        两边都说"她会待着"的时候没有证据，不计分（`rehearsal_trivial`）。
        """
        if self._rehearsal is None:
            return
        done = self.detector.current_attractor
        done = -1 if done is None else int(done)
        self._rehearsal["actual"].append(done)
        self._rehearsal["remaining"] -= 1
        if self._rehearsal["remaining"] > 0:
            return

        start = int(self._rehearsal["start"])
        pred_exit = self._first_exit(self._rehearsal["predicted"], start)
        act_exit = self._first_exit(self._rehearsal["actual"], start)
        self._rehearsal = None

        if pred_exit is None and act_exit is None:
            self.rehearsal_trivial += 1
            return
        hit = pred_exit == act_exit
        self.rehearsal_total += 1
        self.rehearsal_hits += int(hit)
        self.rehearsal_gain.observe(hit)
        self.rehearsal_history.append((bool(hit), pred_exit, act_exit))

    @property
    def exploration_gain(self) -> float:
        """她此刻的探索倍率：一步预测的增益 × 预演的增益。有界（两个都在 [0.8, 1.4]）。"""
        return float(self.self_gain.gain()) * float(self.rehearsal_gain.gain())

    @property
    def rehearsal_accuracy(self) -> float:
        return (self.rehearsal_hits / self.rehearsal_total
                if self.rehearsal_total else 0.0)

    # ---------- 行动 ----------

    def _choose_action(self) -> int:
        """她此刻的动作。强制口只给干预实验用，正常运行时不设。"""
        # 她自己感到的张力（上一步 set_drive 之后留下的那个）。先决定，再记账。
        self._tension_now = float(self.drive.summary().get("sum", 0.0))

        if self.force_action is not None:
            action = int(self.force_action)
        else:
            action = self.actions.choose(self._tension_now, list(self._tension_history))
        self._tension_history.append(self._tension_now)

        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        if action == LEAVE:
            self._apply_leave()
        return action

    def _own_scale(self):
        """她自己量到的区域半径中位数。没有就不推。

        第一次用的是梯度尺度，并且加在速度上 —— 实测 600 步后位移 6323，
        因为每一步都在加速。现在改成按位置挪动、幅度是她的区域半径中位数：
        单次动作有界（一个区域半径），不会累积成火箭。
        """
        attrs = self.landscape.state_space.attractors()
        if not attrs:
            return None
        sigmas = np.asarray([float(a.sigma) for a in attrs], dtype=float)
        sigmas = sigmas[np.isfinite(sigmas) & (sigmas > 0)]
        if sigmas.size == 0:
            return None
        return float(np.median(sigmas))

    def _apply_leave(self) -> None:
        """离开：朝**最近的另一个区域**挪一步，一步最多一个区域半径。

        2026-11-17 改：原来朝「背离质心」的方向挪。当晚实测（副本，真实归属判据
        `距离 < a.sigma`）发现那会让她掉进区域之间的空地——她的半径中位 617、
        最近邻距离中位 717，128 维里球之间的缝隙足够宽，推出去就到不了隔壁。
        于是 `action: hold`、`leave_share: 0.0`：不是她不想走，是走的路通向空地。

        现在把「其它区域的质心」传进去，她朝最近的那个走。
        """
        node = self.detector.current_attractor
        space = self.landscape.state_space
        # The detector publishes a name; the centroid lives in a row. Look the
        # row up now, when it is needed, instead of keeping a second copy of
        # the identity that has to be renumbered after every structural change.
        row = None if node is None else space.uid_row().get(int(node))
        if row is None:
            return
        scale = self._own_scale()
        if scale is None:
            return
        attrs = space.attractors()
        # 除了自己之外的所有质心：她「能去哪」的候选。只有一个区域时没有候选，
        # 那就退回旧行为（背离质心）——没地方可去，至少别崩。
        others = None
        uids = None
        # 2026-11-24：`places` 也必须在「只有一个区域」时存在。原来它只在
        # `len(attrs) > 1` 分支里赋值，于是她只剩一个区域时任何一次 LEAVE 都
        # 抛 UnboundLocalError —— 由 tests/test_no_senescence 的强制 LEAVE 探针抓到。
        # 真机有 1016 个区域所以没暴露；这是一个潜伏的崩溃点。
        places = None
        if len(attrs) > 1:
            idx = [i for i in range(len(attrs)) if i != row]
            others = np.stack([np.asarray(attrs[i].mu, dtype=float) for i in idx])
            # 2026-11-18 夜：**回到位置键**（uid 化回滚）。
            #
            # 人家 23:30 试过用 uid 判断「去过没去过」，看起来多了一个 31323，
            # 但 `test_uid.py::test_consistent_relabelling_changes_nothing` 红了：
            # **把区域重新编号之后她的轨迹变了**——那就违背了「uid 只是名字，不参与决策」。
            # 所以回滚。破环靠的是 action.py 的门槛（dist >= s/2），那里只用距离，不碰 uid。
            # 2026-11-21 21:20：**「去过」从位置键改成 uid 键**。
            # 位置键的问题（persist.py:266-278 的实测）：存下来的位置和现在的质心交集 0，
            # 3868 个位置、4.67 MB、命中率 0——她「以为」所有区域都没去过，于是永远选最近的
            # 两个（卡住）。区域有稳定 uid（Attractor.uid），用它当「去过」的键：
            # 存得住、对得上、也不会因为质心漂移而失效。
            places = [int(getattr(attrs[i], "uid", i)) for i in idx]
            self._visited_places.add(int(getattr(attrs[row], "uid", row)))

        # 2026-11-17 23:05：她「去过哪些」——这就是「探索」策略的凭据。
        #
        # 2026-11-25（A1 候选④）：两个改动。
        # 1. 回家的锚点从「最近的区域」换成「这次外出的出发点」：新长出来的区域
        #    会让「最近」一直跟着她走，原来的 4σ 锚定因此失效（实测最远 8.7 万）。
        # 2. 目标持久化：走到目标半径内才换目标。每步重选的话，一步 151.5
        #    永远只够到最近那个区域（实测只有 2 区、50 步 HOLD 后整段对调）。
        _cur_x = np.asarray(self._st.x, dtype=float)
        _home_dir = None
        if self._home_x is not None:
            _back = self._home_x - _cur_x
            _hd = float(np.linalg.norm(_back))
            if _hd <= float(scale):
                self._home_x = None          # 回到家：这次外出结束
            elif _hd > float(self.home_radius_sigmas) * float(scale):
                self._leave_target = None    # 越界：目标作废，先回家
                _home_dir = _back / _hd * float(scale)
        if self._home_x is None:
            self._home_x = _cur_x.copy()     # 这次外出的出发点
        if _home_dir is None and self._leave_target is not None:
            _trow = space.uid_row().get(int(self._leave_target))
            if _trow is None:
                self._leave_target = None
            else:
                _tmu = np.asarray(attrs[_trow].mu, dtype=float)
                if float(np.linalg.norm(_tmu - _cur_x)) <= float(scale):
                    self._leave_target = None   # 到了，下一次重选

        if _home_dir is not None:
            d = _home_dir
        else:
            d = self.actions.direction(
                self._st.x, attrs[row].mu, scale,
                neighbours=others, visited=self._visited_places,
                places=places,
                prefer_far=(len(set(self._arrival_hist)) <= 3
                            and len(self._arrival_hist) >= 4),
                preferred_target=self._leave_target,
                home=self._home_x if self.filter_targets_by_home else None,
                max_home_dist=(float(self.home_radius_sigmas) * float(scale)
                               if self.filter_targets_by_home else None))
            if d is not None:
                self._leave_target = self.actions.last_target
        if d is not None:
            self._st = DynamicState(x=self._st.x + d, v=self._st.v)
            # 2026-11-21 21:20：到达后记「她到过的区域 uid」（不是位置）。
            # 她落在哪个区域，就记哪个的 uid；落在空地（没归属）时不记——
            # 「去过一片空地」不是「去过一个区域」。
            try:
                _here_node = self.detector.current_attractor
                if _here_node is not None:
                    self._visited_places.add(int(_here_node))
            except Exception:  # noqa: BLE001
                pass

        # ── 锚定（2026-11-18 08:20）──────────────��───────────────
        #
        # 病：_apply_leave 每步直接把 x 挪 617，方向一致时**链式累积**。
        # 实测：她从 11 月 17 日夜里跑到 11 月 18 日早上，|x| 到了 4,091,098，
        # 全部 3197 个区域的质心都在 171 万 - 409 万之间。
        # 结果 `assert_l5_finite` 拒绝保存（上限 1e6），P2 一晚上崩了 4 次。
        #
        # 这不是「探索错了」，是「探索没有边界」：她去没去过的地方没错，
        # 但她不该在**离开所有区域之后**还一直往前走。
        #
        # 修法：每次离开之后，如果她离**最近的区域质心**超过「她自己的半径的 4 倍」，
        # 就把她拉回那个质心上。4 倍是人家选的（可议），但它的含义是
        # 「她可以在区域之间移动，但不该飘到没有区域的地方」。
        try:
            _attrs = space.attractors()
            if _attrs:
                _mus = np.stack([np.asarray(a.mu, dtype=float) for a in _attrs])
                _d = np.linalg.norm(_mus - np.asarray(self._st.x, dtype=float), axis=1)
                _j = int(np.argmin(_d))
                # 2026-11-21 回退：锚定半径还是 4×（人家试过「由她的探索范围定」，
                # 但 _apply_leave 里量到的"离最近区域的距离"一直是 0（她刚挪一步），
                # 于是 _lim 反而收紧到 2×。语义搞错了，回退。
                _lim = 4.0 * float(scale) if scale else 0.0
                if _lim > 0 and float(_d[_j]) > _lim:
                    self._st = DynamicState(x=_mus[_j].copy(), v=self._st.v)
        except Exception:  # noqa: BLE001
            pass

        # 2026-11-21 23:20：改成 uid（和 _visited_places 一致），并且能存档。
        # 原来存的是位置 tuple（`tuple(round(mu,4))`）——和 visited 一样的老问题：
        # 位置会漂、和现状对不上；而且存档时 int(tuple) 直接崩（今晚 23:07 崩了 4 次）。
        # uid 是稳定的区域身份：存得住、比得上是、也不会崩。
        try:
            _arr = getattr(self.detector, "current_attractor", None)
            if _arr is not None:
                self._arrival_hist.append(int(_arr))
        except Exception:  # noqa: BLE001
            pass

    @property
    def action_name(self) -> str:
        return ACTION_NAMES.get(self._action, "hold")

    @property
    def leave_share(self) -> float:
        total = sum(self.action_counts.values())
        return self.action_counts.get(LEAVE, 0) / float(total) if total else 0.0

    # ---------- continuous output ----------

    def broadcast_state(self) -> dict:
        """
        The fields the broadcast actually carries, and nothing else.

        stream_state() computes salient counts, a regulation summary (which
        re-runs the viability measurement over every region) and a drive
        summary. Those are for observation and for a snapshot taken every few
        minutes. The broadcast is sent on every transition, and downwind only
        attractor_id is read: L6 is allowed the id and no continuous quantity.
        Building the full dictionary on every transition did that work for
        nothing.
        """
        st = self._st
        return {
            "attractor_id": (self.detector.current_attractor
                             if self.detector.current_attractor is not None else -1),
            # 2026-11-24 15:10：**GWT 接 L6**（意识层第一步，白默点头的方向）。
            #
            # `workspace.observe()` 每步都在跑（点火 = 全局工作空间广播），
            # 但之前只存在 `_last_ignition` 里、没有任何下游读它——「她注意什么」
            # 和「她说什么」是断的。这里把「点火的那个区域 uid」放进广播，
            # L6 就能读到「她此刻注意什么」。
            #
            # 注意：`ignition` 是「点火」不是「位置」——它是「这一刻全系统可见的东西」，
            # 和 `attractor_id`（她在哪个区域）不是同一个量。
            "ignition": (self._last_ignition.attractor_id
                         if self._last_ignition is not None else -1),
            "potential": (self.landscape.potential(st.x) if st is not None else 0.0),
            # Kept because StreamState carries a vector and the contract says so.
            # It is not read by L6; the whitelist in the adapter is what enforces
            # that, not the absence of the field here.
            "x": st.x.tolist() if st is not None else [],
        }

    def stream_state(self) -> dict:
        st = self._st
        ss = self.landscape.state_space
        return {
            "x": st.x.tolist() if st is not None else [],
            "speed": Dynamics.speed(st) if st is not None else 0.0,
            "attractor_id": (self.detector.current_attractor
                             if self.detector.current_attractor is not None else -1),
            # 2026-11-24：点火也进观测流。之前它只被 L6 读到（-1 一路下去），
            # 日志里查不到「这一步有没有点火」——GWT 因此无法被评估。
            "ignition": (self._last_ignition.attractor_id
                         if self._last_ignition is not None else -1),
            # 2026-11-25：回家规则的可观测性（A1 候选④）。
            "leave_target": (None if self._leave_target is None
                             else int(self._leave_target)),
            "home_dist": (None if self._home_x is None
                          else round(float(np.linalg.norm(
                              self._home_x - self._st.x)), 1)),
            # 2026-11-27：最近 512 步的概率事件计数——安静时看是 sustain 还是 below_ref。
            "gwt_reasons": dict(collections.Counter(
                r for r in self._gwt_reasons if r)),
            # 2026-11-26：AST（只读）——点火预测的 bits 与样本数。
            "attn_n": int(self.attention.observations),
            "attn_acc": round(float(self.attention.accuracy), 4),
            "attn_bits": round(float(self.attention.bits_per_ignition), 4),
            # 2026-11-27：最近 256 次点火的 bits——累计值会把退化平均掉。
            "attn_bits_recent": round(float(self.attention.recent_bits), 4),
            "attn_base_bits": round(
                float(self.attention.base_bits_per_ignition), 4),
            "attn_conf": round(float(self.attention.mean_confidence), 4),
            "potential": (self.landscape.potential(st.x) if st is not None else 0.0),
            "n_attractors": len(ss),
            # 行动是她自己的，不是从外面看出来的统计量：记名字与比例。
            "action": ACTION_NAMES.get(self._action, "hold"),
            "leave_share": round(self.leave_share, 4),
            "tension_now": round(float(self._tension_now), 5),
            "exploration_gain": round(self.exploration_gain, 4),
            "rehearsal_accuracy": round(self.rehearsal_accuracy, 4),
            "rehearsals": int(self.rehearsal_total),
            "rehearsals_trivial": int(self.rehearsal_trivial),
            "n_salient": len(ss.salient()),
            "global_scale": ss.global_scale,
            "viability": self.regulation.summary(ss),            # L3
            "n_edges": len(self.graph),
            "recall": (list(self._last_recall.path)
                       if self._last_recall is not None else []),
            # L4
            "tensions": self.drive.summary(),
            # L8
            "identity": (len(self._last_core.edges)
                         if self._last_core is not None else 0),
        }

    # Compatibility alias for the previous field names.
    def stream_state_legacy(self) -> dict:
        d = self.stream_state()
        return {
            "x": d["x"],
            "speed": d["speed"],
            "attractor_id": d["attractor_id"],
            "potential": d["potential"],
            "n_traces": d["n_attractors"],
            "n_basins": d["n_salient"],
        }

    # ---------- discrete output ----------

    def get_symbols(self) -> SymbolBundle:
        """The only thing L6 receives. No vectors, no continuous quantities."""
        aid = self.detector.current_attractor
        seq = self.detector.recent_sequence(k=8)
        return SymbolBundle(
            attractor_id=str(aid) if aid is not None else "none",
            transition_seq=[str(s) for s in seq],
            intent="state" if aid is not None else "silent",
            t_event=datetime.now(timezone.utc).isoformat(),
            # A label, not a number: the region's most usual position in her
            # own measured context, or "none" when that is not yet measurable.
            referent=(self.grounding.label(int(aid))
                      if aid is not None else "none"),
        )
