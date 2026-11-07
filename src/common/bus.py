"""
src/common/bus.py — 通信总线

依据：
- CONTRACTS/00_overview.md（端口、topic、订阅白名单、缓冲策略）
- ARCHITECTURE.md §2（层间通信）、§4.3（LLM 输出不回流）

红线（代码即法律）：
1. publish() 只接受带 to_json() 的对象，拒绝裸 dict
2. subscribe() 根据调用方模块名 + 白名单检查，违规抛 ArchitectureViolation
3. 任何内部进程订阅 l6.out 抛 ArchitectureViolation
4. 不存在同步屏障：旧版的 5564 ACK 机制已于 2026-11-01 废除（见 CONTRACTS §2、§4）
5. 每个进程定时写 data/logs/<process>.heartbeat —— 属于工程标度（B 类），不是认知节律
"""

from __future__ import annotations

import inspect
import queue
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import zmq

from src.common.types import ArchitectureViolation, SchemaMismatch


# ============================================================
# 常量（与 00_overview.md §5 一一对应）
# ============================================================

PROCESS_PUB_PORT = {"P1": 5561, "P2": 5562, "P3": 5563, "P4": 5560}

TOPIC_PREFIX_TO_PORT = {
    "l0.": 5561,
    "l1.": 5561,
    # 过渡期例外（2026-11-03）：L2 目前住在 P1 里，所以它的话题必须从 5561 发出。
    # 按架构 L2 应该在 P2 内部，那时它会回到 5562。
    # 拆分 L2 到 process_2_mind.py 之后，把 "l2." 改回 5562。
    "l2.": 5561,
    "l4.": 5562,
    "l5.": 5562,
    "l3.": 5563,
    "l7.": 5563,
    "l8.": 5563,
    "l6.": 5560,
}

# 订阅白名单：进程 → 允许的 topic 前缀；None 表示不限制（EXTERNAL）
# 2026-11-01 重写：l7.broadcast → l7.event；P1 新增只能观察 l7.event.phase
SUBSCRIBE_WHITELIST: Dict[str, Optional[List[str]]] = {
    "P1": ["l7.event.phase", "l6.text.signal"],
    # P2 的 l2. 前缀是**过渡期例外**：按架构 L2 应该在 P2 内部，
    # 不跨进程订阅。但 L2 目前还在 P1 中，拆进程完成前需要读取它的广播。
    # 拆完后删除 "l2."。
    "P2": ["l0.", "l1.", "l2.state.field", "l3.", "l7.", "l8.state"],
    "P3": ["l0.", "l1.state.surprise", "l2.", "l4.", "l5."],
    "P4": ["l5.state.stream", "l2.state", "l4.state", "l7.event"],
    "EXTERNAL": None,
}

# 缓冲默认策略：topic → (kind, capacity)
# 与 CONTRACTS/00_overview.md §8.3 一一对应。
# 不存在 "sync" 标记：同步屏障已废除。
BUFFER_DEFAULTS: Dict[str, tuple] = {
    "l6.text.signal": ("queue", 16),
    "l0.event.envelope": ("queue", 8),
    "l1.state.surprise": ("latest", 0),
    "l1.state.projection": ("latest", 0),
    "l2.state.field": ("latest", 0),
    "l4.state.drive": ("latest", 0),
    "l5.state.stream": ("latest", 0),
    "l5.event.transition": ("queue", 32),
    "l3.state.activation": ("latest", 0),
    "l3.event.landscape_update": ("queue", 8),
    "l7.event.phase": ("queue", 4),
    "l7.event.stage": ("latest", 0),
    "l8.state.selfmodel": ("latest", 0),
    "l8.event.chain_node": ("queue", 128),
    "l6.out": ("latest", 0),
}

HEARTBEAT_INTERVAL = 5.0       # 秒；工程标度（B 类），不是认知节律


# ============================================================
# 内部日志（暂不依赖 src/common/logging.py）
# ============================================================

_LOG_LOCK = threading.Lock()


def _log(process_id: str, level: str, msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} [{level}] {process_id} {msg}\n"
    with _LOG_LOCK:
        try:
            path = Path("data/logs/bus.log")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass  # 日志失败不能拖垮总线


# ============================================================
# 缓冲
# ============================================================

class LatestBuffer:
    """覆盖式：新消息覆盖旧消息，只保留最新。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Any = None
        self._has = threading.Event()

    def put(self, item: Any) -> None:
        with self._lock:
            self._item = item
        self._has.set()

    def get(self, timeout: float = 0.1) -> Any:
        if not self._has.wait(timeout):
            return None
        with self._lock:
            item = self._item
            self._item = None
            self._has.clear()
        return item


class QueueBuffer:
    """队列式：满时丢最旧。"""

    def __init__(self, capacity: int) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=capacity)

    def put(self, item: Any) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()  # 丢最旧
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:
                pass

    def get(self, timeout: float = 0.1) -> Any:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


# ============================================================
# Subscription：一个 topic 的消费线程
# ============================================================

class Subscription:
    def __init__(
        self,
        topic: str,
        buffer_kind: str,
        capacity: int,
        handler: Callable[[str, Any], None],
        parser: Callable[[str], Any],
        bus: "Bus",
    ) -> None:
        self.topic = topic
        self.buffer_kind = buffer_kind
        self.handler = handler
        self.parser = parser
        self.bus = bus
        if buffer_kind == "latest":
            self.buffer: Any = LatestBuffer()
        elif buffer_kind == "queue":
            self.buffer = QueueBuffer(capacity)
        else:
            raise ValueError(f"unsupported buffer_kind: {buffer_kind}")
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"sub-{topic}"
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self.buffer.get(timeout=0.1)
            if item is None:
                continue
            topic, raw_payload = item
            try:
                obj = self.parser(raw_payload)
            except SchemaMismatch as e:
                _log(self.bus.process_id, "WARN", f"schema_mismatch topic={topic} {e}")
                continue
            except Exception as e:
                _log(self.bus.process_id, "ERROR", f"parse_error topic={topic} {e!r}")
                continue
            try:
                self.handler(topic, obj)
            except Exception:
                _log(
                    self.bus.process_id,
                    "ERROR",
                    f"handler_error topic={topic} traceback={traceback.format_exc()}",
                )

    def stop(self) -> None:
        self._stop.set()


# ============================================================
# 模块名检测（红线强制用）
# ============================================================

def _caller_module() -> str:
    """返回调用 subscribe / publish 的模块全名，如 'src.L5_cognition.stream'。"""
    for frame_info in inspect.stack()[2:]:  # 跳过 _caller_module 自身和 subscribe
        name = frame_info.frame.f_globals.get("__name__", "")
        if name and not name.startswith("src.common.bus"):
            return name
    return "unknown"


def _check_whitelist(process_id: str, topic: str, caller_mod: str) -> None:
    if process_id == "EXTERNAL":
        return
    allowed = SUBSCRIBE_WHITELIST.get(process_id)
    if allowed is None:
        return
    if not any(topic.startswith(prefix) for prefix in allowed):
        raise ArchitectureViolation(
            f"process={process_id} module={caller_mod} cannot subscribe topic={topic!r}; "
            f"whitelist={allowed}"
        )


# ============================================================
# Bus 主类
# ============================================================

class Bus:
    def __init__(self, process_id: str) -> None:
        if process_id not in ("P1", "P2", "P3", "P4", "EXTERNAL"):
            raise ValueError(f"unknown process_id: {process_id}")
        self.process_id = process_id
        self._context = zmq.Context.instance()

        # PUB：只有非 EXTERNAL 进程绑定
        self._pub_socket: Optional[zmq.Socket] = None
        if process_id != "EXTERNAL":
            self._pub_socket = self._context.socket(zmq.PUB)
            self._pub_socket.setsockopt(zmq.LINGER, 0)
            port = PROCESS_PUB_PORT[process_id]
            self._pub_socket.bind(f"tcp://*:{port}")
            _log(process_id, "INFO", f"PUB bound tcp://*:{port}")

        # SUB：按端口索引
        self._sub_sockets: Dict[int, zmq.Socket] = {}
        self._subscriptions: Dict[str, List[Subscription]] = {}
        self._sub_lock = threading.Lock()
        self._recv_threads: List[threading.Thread] = []
        self._stop = threading.Event()

        # 心跳
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        )
        self._heartbeat_thread.start()

    # ---------- PUB ----------

    def publish(self, topic: str, obj: Any) -> None:
        """发布一条消息。obj 必须有 to_json()。"""
        if self._pub_socket is None:
            raise ArchitectureViolation(
                f"process={self.process_id} has no PUB socket (EXTERNAL cannot publish)"
            )
        if not hasattr(obj, "to_json"):
            raise ArchitectureViolation(
                f"publish() only accepts objects with to_json(), got {type(obj).__name__}"
            )
        # 校验 topic 归属
        port = self._topic_port(topic)
        if PROCESS_PUB_PORT[self.process_id] != port:
            raise ArchitectureViolation(
                f"process={self.process_id} cannot publish topic={topic!r} "
                f"(belongs to port {port})"
            )
        self._pub_socket.send_multipart([topic.encode("utf-8"), obj.to_json().encode("utf-8")])

    # ---------- SUB ----------

    def subscribe(
        self,
        topic: str,
        handler: Callable[[str, Any], None],
        parser: Optional[Callable[[str], Any]] = None,
        buffer_kind: Optional[str] = None,
        capacity: int = 0,
        owner_module: Optional[str] = None,
    ) -> Subscription:
        """
        topic: 前缀或完整 topic（ZeroMQ SUB 是前缀匹配）
        handler: (topic, obj) -> None
        parser: raw_json -> obj；若为 None 则传 str 给 handler
        buffer_kind: 'latest' / 'queue'；None 时按 BUFFER_DEFAULTS
        capacity: 仅在 buffer_kind == 'queue' 时有效
        owner_module: 测试用 override；生产环境不要传，走自动检测
        """
        caller_mod = owner_module or _caller_module()
        _check_whitelist(self.process_id, topic, caller_mod)

        # 选 buffer 策略
        if buffer_kind is None:
            default = self._buffer_default(topic)
            buffer_kind = default[0]
            if capacity == 0:
                capacity = default[1]
            if buffer_kind not in ("latest", "queue"):
                raise ValueError(f"unsupported buffer_kind: {buffer_kind}")

        # 确定端口并确保 SUB socket 存在
        port = self._topic_port(topic)
        with self._sub_lock:
            if port not in self._sub_sockets:
                sock = self._context.socket(zmq.SUB)
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(f"tcp://127.0.0.1:{port}")
                # 关键：SUBSCRIBE 必须在 recv 线程启动之前设置。
                # libzmq 的 socket 不是线程安全的——recv 线程开始 poll 后，
                # 主线程再 setsockopt 会导致订阅不生效，消息全被丢弃。
                sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
                self._sub_sockets[port] = sock
                t = threading.Thread(
                    target=self._recv_loop,
                    args=(port, sock),
                    daemon=True,
                    name=f"recv-{port}",
                )
                t.start()
                self._recv_threads.append(t)
                _log(self.process_id, "INFO", f"SUB connected tcp://127.0.0.1:{port}")
            else:
                # 已有 socket。同一进程启动后一般只 subscribe 一次；
                # 若将来需要动态追加 topic，这里仍有竞态，届时需要重构为
                # 每个 subscription 独立 socket。
                sock = self._sub_sockets[port]
                sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))

        parser_fn = parser or (lambda s: s)
        sub = Subscription(topic, buffer_kind, capacity, handler, parser_fn, self)
        with self._sub_lock:
            self._subscriptions.setdefault(topic, []).append(sub)
        _log(self.process_id, "INFO", f"subscribe topic={topic!r} buffer={buffer_kind}")
        return sub

    def _recv_loop(self, port: int, sock: zmq.Socket) -> None:
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                events = dict(poller.poll(timeout=100))
            except zmq.ContextTerminated:
                return
            if sock not in events:
                continue
            try:
                parts = sock.recv_multipart()
            except zmq.ContextTerminated:
                return
            if len(parts) < 2:
                continue
            topic = parts[0].decode("utf-8")
            payload = parts[1].decode("utf-8")

            # 分发给匹配的 subscription
            with self._sub_lock:
                subs = [
                    sub
                    for t, sublist in self._subscriptions.items()
                    for sub in sublist
                    if topic.startswith(t)
                ]
            for sub in subs:
                sub.buffer.put((topic, payload))

    # ---------- 辅助 ----------

    def _topic_port(self, topic: str) -> int:
        for prefix, port in TOPIC_PREFIX_TO_PORT.items():
            if topic.startswith(prefix):
                return port
        raise ArchitectureViolation(
            f"topic={topic!r} does not match any known prefix "
            f"{list(TOPIC_PREFIX_TO_PORT.keys())}"
        )

    def _buffer_default(self, topic: str) -> tuple:
        # 精确匹配，其次前缀匹配，最后回退 latest
        if topic in BUFFER_DEFAULTS:
            return BUFFER_DEFAULTS[topic]
        for t, cfg in BUFFER_DEFAULTS.items():
            if topic.startswith(t):
                return cfg
        return ("latest", 0)

    def list_subscriptions(self) -> List[Dict[str, str]]:
        with self._sub_lock:
            return [
                {
                    "topic": sub.topic,
                    "buffer": sub.buffer_kind,
                    "handler": getattr(sub.handler, "__qualname__", repr(sub.handler)),
                }
                for subs in self._subscriptions.values()
                for sub in subs
            ]

    def _heartbeat_loop(self) -> None:
        if self.process_id == "EXTERNAL":
            return
        path = Path("data/logs") / f"{self.process_id}.heartbeat"
        path.parent.mkdir(parents=True, exist_ok=True)
        while not self._stop.is_set():
            try:
                path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
            except Exception:
                pass
            self._stop.wait(HEARTBEAT_INTERVAL)

    # ---------- 停止 ----------

    def stop(self) -> None:
        _log(self.process_id, "INFO", "stopping")
        self._stop.set()
        for subs in self._subscriptions.values():
            for sub in subs:
                sub.stop()
        for t in self._recv_threads:
            t.join(timeout=1.0)
        for sock in self._sub_sockets.values():
            try:
                sock.close(linger=0)
            except Exception:
                pass
        if self._pub_socket is not None:
            try:
                self._pub_socket.close(linger=0)
            except Exception:
                pass