r"""领地授权：一条可判定的不变量，而不是"加了把锁"。

不变量：外部 PUB 允许集合 ∩ 认知层消费的 topic 集合 = {l6.text.signal}。
另外验证 CURVE 真的能拦住无密钥的客户端（真实 ZMQ 握手，不是纸面）。

Run: .\.venv\Scripts\python.exe tests\test_endpoint.py
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.common import endpoint  # noqa: E402


def test_only_the_input_topic_is_externally_writable() -> None:
    writable = {t for t in endpoint.COGNITIVE_TOPICS
                if endpoint.authorize("sensor", t, "pub")}
    assert writable == {endpoint.INPUT_TOPIC}, writable


def test_sensor_cannot_write_cognitive_state() -> None:
    for topic in ("l2.state.field", "l5.event.transition", "l1.state.error",
                  "l0.event.envelope"):
        assert not endpoint.authorize("sensor", topic, "pub"), topic
        try:
            endpoint.require("sensor", topic, "pub")
            raise AssertionError(f"{topic} accepted")
        except endpoint.Denied:
            pass


def test_actuator_cannot_publish_anything() -> None:
    assert not endpoint.authorize("actuator", endpoint.INPUT_TOPIC, "pub")
    assert endpoint.authorize("actuator", "l5.state.stream", "sub")
    assert not endpoint.authorize("actuator", "l2.state.field", "sub")


def test_unknown_role_and_direction_denied() -> None:
    assert not endpoint.authorize("root", endpoint.INPUT_TOPIC, "pub")
    assert not endpoint.authorize("sensor", endpoint.INPUT_TOPIC, "write")


def test_rate_limiter_drops_instead_of_queueing() -> None:
    rl = endpoint.RateLimiter(capacity=3, refill_per_s=0)
    allowed = sum(1 for _ in range(10) if rl.allow(now=100.0))
    assert allowed == 3, allowed
    rl2 = endpoint.RateLimiter(capacity=1, refill_per_s=10)
    assert rl2.allow(now=0.0)
    assert not rl2.allow(now=0.0)
    assert rl2.allow(now=0.2)


def test_curve_transport_blocks_peers_without_the_server_key() -> None:
    """真实 ZMQ/CURVE：拿到 server 公钥的客户端能收到；拿不到的一个字节也收不到。

    客户端白名单（按公钥精确授权）需要 ZAP authenticator，属接线步骤（见交接文档），
    这里只验证传输层不认匿名对端。
    """
    import zmq

    server_pub, server_sec = endpoint.curve_keypair()
    good_pub, good_sec = endpoint.curve_keypair()
    imposter_pub, imposter_sec = endpoint.curve_keypair()
    wrong_server_pub, _ = endpoint.curve_keypair()

    keys = endpoint.ServerKeys(server_pub, server_sec, frozenset({good_pub}))
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    keys.configure(pub)                       # 必须在 bind 之前
    port = pub.bind_to_random_port("tcp://127.0.0.1")

    good = endpoint.client_socket(ctx, server_public=server_pub,
                                  client_public=good_pub, client_secret=good_sec)
    good.setsockopt(zmq.SUBSCRIBE, b"")
    good.setsockopt(zmq.RCVTIMEO, 4000)
    good.connect(f"tcp://127.0.0.1:{port}")

    imposter = endpoint.client_socket(ctx, server_public=wrong_server_pub,
                                      client_public=imposter_pub, client_secret=imposter_sec)
    imposter.setsockopt(zmq.SUBSCRIBE, b"")
    imposter.setsockopt(zmq.RCVTIMEO, 1500)
    imposter.connect(f"tcp://127.0.0.1:{port}")

    time.sleep(1.0)
    for _ in range(20):
        pub.send(b"hello")
        time.sleep(0.05)

    try:
        assert good.recv() == b"hello", "authorised client must receive"
        try:
            imposter.recv()
            raise AssertionError("a peer without the server key received a payload")
        except zmq.Again:
            pass
    finally:
        for s_ in (good, imposter, pub):
            s_.close(0)
        ctx.term()

    # 策略层：白名单非空才允许，空集合 = 全拒
    assert keys.permits(good_pub) and not keys.permits(imposter_pub)
    assert not endpoint.ServerKeys(server_pub, server_sec).permits(good_pub)
    print("       CURVE: authorised received; peer without server key timed out")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())