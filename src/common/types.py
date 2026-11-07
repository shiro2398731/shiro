"""
src/common/types.py

所有跨层数据类的唯一定义处。契约（CONTRACTS/*.md）里的每一个 schema，
在这里必须有一一对应的 dataclass。禁止在别的模块里重新定义同一语义的类。

序列化规则：
- 所有跨进程 payload 必须通过 .to_json() / .from_json() 序列化
- 禁止手写 json.dumps / json.loads 来传递跨层数据
- to_json() 返回 str（UTF-8 可编码），from_json() 返回类实例
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional


# ============================================================
# 异常
# ============================================================

class ArchitectureViolation(Exception):
    """当某层违反 ARCHITECTURE.md 的通信或 LLM 边界原则时抛出。"""
    pass


class SchemaMismatch(Exception):
    """当 from_json() 的 payload 与 dataclass 字段不匹配时抛出。"""
    pass


# ============================================================
# 时间与时钟
# ============================================================

Tier = Literal["T0", "T1", "T2", "T3", "T4", "T5"]


@dataclass(frozen=True)
class Tick:
    """src/common/clock.py 广播的时钟节拍。"""
    tier: Tier
    timestamp: str          # ISO 8601
    seq: int                # 该 tier 的单调递增序号

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Tick":
        return cls(**json.loads(s))


# ============================================================
# L0 → 全体：包络事件（相位的唯一物理来源）
# ============================================================

Phase = Literal["up", "down", "unknown"]


@dataclass(frozen=True)
class EnvelopeEvent:
    """
    l0.event.envelope 的 payload。

    这是全系统唯一的相位来源。相位不是被广播出来的命令，而是 L0 慢包络的
    实测状态被说出来。L7 只能观察它，不能生产者它（见 CONTRACTS §6）。

    period_s 是该时刻对 L0 完整循环周期的估计，由最近若干次上穿间隔的中位数
    算出。它是个滑动量，不是常数。为 None 表示样本还不够，尚未建立估计。
    """
    phase: Phase
    period_s: Optional[float]    # 实测周期估计，秒；None = 尚未建立
    t_event: str                 # 事件发生的墙钟时刻，ISO 8601 微秒精度

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "EnvelopeEvent":
        return cls(**json.loads(s))


# ============================================================
# L1 → 上层：惊讶与投影
# ============================================================

@dataclass(frozen=True)
class SurpriseSignal:
    """
    CONTRACTS/L1_L2.md 定义的惊讶信号。
    注意：s_t 是标量，channel_errors 是通道分解，precision 是精度加权向量。
    """
    s_t: float                          # 标量惊讶，>= 0
    channel_errors: Dict[str, float]    # 各通道的预测误差
    precision: Dict[str, float]         # 各通道的精度权重，和为 1.0
    timestamp: str
    t_event: str = ""                   # 事件真实发生的墙钟时刻（CONTRACTS §7 要求）

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "SurpriseSignal":
        return cls(**json.loads(s))


@dataclass(frozen=True)
class ProjectionVector:
    """
    L0 → L1 的三层投影，供 L1 的编码器消费。
    三个向量长度固定，由 config/time_scales.toml 决定。
    """
    fast: List[float]       # 快层投影，维度见 config
    mid: List[float]        # 中层投影
    slow: List[float]       # 慢层投影
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "ProjectionVector":
        return cls(**json.loads(s))


# ============================================================
# L2 → 上层：情感
# ============================================================

ValenceSign = Literal["+", "-", "0"]
Magnitude = Literal["low", "mid", "high"]
Dimension = Literal[
    "curious", "calm", "uneasy", "bored", "close", "distant", "neutral"
]


@dataclass(frozen=True)
class AffectTag:
    """CONTRACTS/L5_L6.md §2.3 的 affect_tag 结构。"""
    valence_sign: ValenceSign
    valence_mag: Magnitude
    arousal_band: Magnitude
    dimension: Dimension

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "AffectTag":
        return cls(**json.loads(s))


@dataclass(frozen=True)
class AffectState:
    """L2 广播的完整情感状态（含连续量，供 L3/L4 消费，不供 L6）。"""
    valence: float          # [-1, +1]
    arousal: float          # [0, 1]
    desire: float           # [0, 1]
    satiety: float          # [0, 1]
    surprise: float         # [0, 1]（来自 L1 的 s_t 归一化）
    ennui: float            # [0, 1]
    tag: AffectTag
    timestamp: str

    def to_json(self) -> str:
        d = asdict(self)
        d["tag"] = asdict(self.tag)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "AffectState":
        d = json.loads(s)
        d["tag"] = AffectTag(**d["tag"])
        return cls(**d)


# ============================================================
# L2 → 上层：内部状态场
# ============================================================

@dataclass(frozen=True)
class StateVector:
    """
    l2.state.field 的 payload。

    16 维功能状态场。每一维一个可计算的功能性量，**没有人类情绪词**。
    维度语义见 src/runtime/process_1_fast.py 的 L2StateField 文档。

    dev 是该点相对她自身历史分布的马氏距离（不是到某个写死原点的距离）。
    """
    v: List[float]              # len == 16
    dev: float
    l2_updates: int             # 累计重算次数（观测用）
    t_event: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "StateVector":
        return cls(**json.loads(s))


# ============================================================
# L4 → 上层：驱动
# ============================================================

DriveName = Literal["survival", "curiosity", "mastery", "connection", "rest"]


@dataclass(frozen=True)
class DriveState:
    """L4 广播的驱动力状态。"""
    weights: Dict[DriveName, float]     # 每条驱动力的当前权重，和为 1.0
    dominant: DriveName                 # argmax
    deflection: List[float]             # 偏转场 a_t，维度见 config
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "DriveState":
        return cls(**json.loads(s))


# ============================================================
# L5 → L6 / L3：符号与跳盆
# ============================================================

Intent = Literal["state", "recall", "query", "silent"]


@dataclass(frozen=True)
class SymbolBundle:
    """
    CONTRACTS/L5_L6.md §2 定义的唯一符号集。
    L6 只能通过 stream.get_symbols() 拿到这个对象，拿不到任何向量。
    """
    attractor_id: str                   # <= 32 chars, [a-z0-9_], or "none"
    transition_seq: List[str]           # <= 8 attractor_id values
    affect_tag: AffectTag
    drive_tag: DriveName
    intent: Intent

    def to_json(self) -> str:
        d = asdict(self)
        d["affect_tag"] = asdict(self.affect_tag)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "SymbolBundle":
        d = json.loads(s)
        d["affect_tag"] = AffectTag(**d["affect_tag"])
        return cls(**d)


@dataclass(frozen=True)
class TextSignalEvent:
    """
    l6.text.signal payload: a sentence already encoded into numbers.

    This is the only path by which language enters the system. It carries no
    interpretation -- just the vector, a few structural tags, and the original
    text for logging. Consumers treat it as an input to be predicted, not as a
    conclusion.
    """
    vector: List[float]
    tags: dict
    text: str
    encoder: str                        # "llm" or "hash"
    t_event: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "TextSignalEvent":
        return cls(**json.loads(s))


@dataclass(frozen=True)
class StreamState:
    """
    L5 广播的连续状态（供 L3 情节编码、L4 反馈，不供 L6）。
    注意：这个对象含向量，L6 不允许订阅 l5.state.stream 来读它。
    L6 只能读 l5.state.stream 的 topic 存在性（用于 gate 判断），
    实际符号通过 stream.get_symbols() 拉取。
    """
    x_t: List[float]                    # current state vector
    attractor_id: str                   # id of the nearest salient attractor
    landscape_energy: float             # potential at the current state
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "StreamState":
        return cls(**json.loads(s))


@dataclass(frozen=True)
class TransitionEvent:
    """L5 attractor transition."""
    from_attractor: str
    to_attractor: str
    timestamp: str
    # True when the transition was not carried by the trajectory
    noise_driven: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "TransitionEvent":
        return cls(**json.loads(s))


# ============================================================
# L3 → 上层：记忆激活与景观更新
# ============================================================

@dataclass(frozen=True)
class ActivationState:
    """L3 广播的当前激活情节。"""
    episode_id: str
    episode_vector: List[float]
    affect_tag: AffectTag
    surprise_peak: float
    timestamp: str

    def to_json(self) -> str:
        d = asdict(self)
        d["affect_tag"] = asdict(self.affect_tag)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "ActivationState":
        d = json.loads(s)
        d["affect_tag"] = AffectTag(**d["affect_tag"])
        return cls(**d)


@dataclass(frozen=True)
class LandscapeUpdate:
    """L3 broadcast: an attractor was created or its depth changed."""
    attractor_id: str
    position: List[float]
    depth: float
    reason: Literal["new_semantic", "consolidated", "decayed"]
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "LandscapeUpdate":
        return cls(**json.loads(s))


# ============================================================
# L7 → 所有层：相位与阶段广播
# ============================================================

Phase = Literal["awake", "sleep"]


@dataclass(frozen=True)
class PhaseBroadcast:
    """
    l7.broadcast.phase 的 payload。
    注意：这是同步屏障，payload 极简，不走复杂数据类。
    """
    phase: Phase
    ts: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "PhaseBroadcast":
        return cls(**json.loads(s))


@dataclass(frozen=True)
class StageBroadcast:
    """l7.broadcast.stage 的 payload。"""
    stage: int                          # 0..4
    name: str                           # "newborn" / "infant" / "toddler" / "child" / "mature"
    ts: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "StageBroadcast":
        return cls(**json.loads(s))


# ============================================================
# L8 → 上层：身份
# ============================================================

@dataclass(frozen=True)
class SelfModelState:
    """L8 广播的自我模型（低维向量，用于 L5 的弱偏转场）。"""
    self_vector: List[float]
    consistency_score: float            # [0, 1]，与最近 K 个节点的连续性
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "SelfModelState":
        return cls(**json.loads(s))


@dataclass(frozen=True)
class ChainNodeEvent:
    """L8 因果链的新节点事件。"""
    node_id: str
    event_type: Literal[
        "transition", "affect_shift", "sleep_wake", "external_input", "homeostasis"
    ]
    snapshot: List[float]               # 低维压缩快照
    parents: List[str]                  # 前驱节点 ID
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "ChainNodeEvent":
        return cls(**json.loads(s))


# ============================================================
# L6 → 外部：表达输出
# ============================================================

@dataclass(frozen=True)
class ExpressionOutput:
    """
    L6 的输出。注意：这个对象只被外部消费者读取，任何内部层不得订阅。
    CONTRACTS/L6_OUTPUT.md 的红线在这里体现为：没有内部模块 import 这个类。
    """
    text: str
    mode: Literal["template", "template_fallback", "llm", "llm_fallback"]
    attractor_id: str
    intent: Intent
    drive_tag: DriveName
    affect_tag: AffectTag
    timestamp: str

    def to_json(self) -> str:
        d = asdict(self)
        d["affect_tag"] = asdict(self.affect_tag)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "ExpressionOutput":
        d = json.loads(s)
        d["affect_tag"] = AffectTag(**d["affect_tag"])
        return cls(**d)


# ============================================================
# 工具：安全的 from_json
# ============================================================

def safe_from_json(cls, s: str):
    """
    包装 from_json，把 KeyError / TypeError 转成 SchemaMismatch，
    附带原始 payload 的前 200 字符，便于调试。
    """
    try:
        return cls.from_json(s)
    except (KeyError, TypeError, ValueError) as e:
        raise SchemaMismatch(
            f"{cls.__name__}.from_json failed: {e!r}; payload[:200]={s[:200]!r}"
        ) from e