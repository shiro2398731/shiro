"""
src/common/endpoint.py

领地（endpoint）授权策略 + CURVE 传输参数。

问题定义
--------
把总线开放到网络，等于把她的输入通路开放给任何人。授权不是"加把锁"，
而是一条可判定的不变量：

    PUB 允许集合 ∩ （被 L1–L5 消费的 topic 集合） = {l6.text.signal}

即：外部只能写"感知入口"这一个 topic，且必须通过 schema 与限额；
其余一切认知层消费的 topic 一律拒绝 PUB，不降级、不兜底。

角色
----
sensor   ：只允许 PUB l6.text.signal；不允许 SUB 任何 topic。
actuator ：只允许 SUB {l5.state.stream, l6.out, l7.event.phase}；不允许 PUB。

本模块只做策略与密钥，不改 bus.py；接线是下一步（见交接文档）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

# 被认知层消费的 topic（= 可以改变她的内部状态）。外部 PUB 必须被拒。
COGNITIVE_TOPICS: frozenset[str] = frozenset({
    "l0.event.envelope",
    "l1.state.prediction",
    "l1.state.error",
    "l2.state.field",
    "l3.event.landscape_update",
    "l4.event.drive",
    "l5.event.transition",
    "l5.state.stream",
    "l6.text.signal",
})

# 唯一允许外部写入的 topic（感知入口）。
INPUT_TOPIC = "l6.text.signal"

ROLES: dict[str, dict[str, frozenset[str]]] = {
    "sensor": {"pub": frozenset({INPUT_TOPIC}), "sub": frozenset()},
    "actuator": {
        "pub": frozenset(),
        "sub": frozenset({"l5.state.stream", "l6.out", "l7.event.phase"}),
    },
}


class Denied(PermissionError):
    """授权失败。调用方必须断开连接，不得降级为匿名。"""


def authorize(role: str, topic: str, direction: str) -> bool:
    """(role, topic, direction) 是否允许。direction ∈ {"pub","sub"}。"""
    spec = ROLES.get(role)
    if spec is None:
        return False
    if direction not in ("pub", "sub"):
        return False
    if direction == "pub" and topic in COGNITIVE_TOPICS and topic != INPUT_TOPIC:
        # 硬约束：认知层消费的 topic 不接受任何外部写入
        return False
    return topic in spec[direction]


def require(role: str, topic: str, direction: str) -> None:
    if not authorize(role, topic, direction):
        raise Denied(f"role={role} {direction} {topic!r} is not permitted")


@dataclass
class RateLimiter:
    """token bucket。感知入口限流：超出即丢弃，不排队（排队会变成背压攻击面）。"""

    capacity: float = 5.0
    refill_per_s: float = 2.0
    _tokens: float = field(default=0.0, repr=False)
    # None until the first call: 初始化时不能读真实时钟，否则注入 now 的调用方
    # 会算出 (now - monotonic) 的巨额负值，第一次 allow 必然失败。
    _last: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._tokens = self.capacity

    def allow(self, cost: float = 1.0, now: float | None = None) -> bool:
        t = time.monotonic() if now is None else now
        if self._last is None:
            self._last = t
        self._tokens = min(self.capacity, self._tokens + (t - self._last) * self.refill_per_s)
        self._last = t
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False


# ---------------------------------------------------------------- CURVE 密钥

def curve_keypair() -> tuple[str, str]:
    """(public, secret)，Z85 编码。需要 pyzmq。"""
    import zmq
    pub, sec = zmq.curve_keypair()
    return pub.decode(), sec.decode()


@dataclass
class ServerKeys:
    """server 侧的 CURVE 参数：自身密钥 + 允许的客户端公钥集合。"""

    public: str
    secret: str
    allowed_clients: frozenset[str] = frozenset()

    def configure(self, socket) -> None:
        """把 CURVE 参数应用到 socket。**必须在 bind/connect 之前调用**。

        ZMQ 在 bind 时固化安全选项：bind 之后再设 curve_server，握手不会启用，
        表现是对端能连上但收不到任何消息（本次实测踩过）。
        """
        import zmq
        socket.curve_server = True
        socket.curve_secretkey = self.secret.encode()
        socket.curve_publickey = self.public.encode()

    def permits(self, client_public: str | Iterable[str]) -> bool:
        if not self.allowed_clients:
            return False                      # 空集合 = 谁都不许（安全默认）
        if isinstance(client_public, str):
            return client_public in self.allowed_clients
        return all(k in self.allowed_clients for k in client_public)


def client_socket(context, *, server_public: str, client_public: str,
                  client_secret: str, socket_type=None):
    """按 CURVE 参数建立客户端 socket（未 connect）。"""
    import zmq
    sock = context.socket(socket_type if socket_type is not None else zmq.SUB)
    sock.curve_secretkey = client_secret.encode()
    sock.curve_publickey = client_public.encode()
    sock.curve_serverkey = server_public.encode()
    return sock