r"""AST 最小实现（attention_schema）：只读的「注意力预测器」。

判据用前向 bits（和 self_model 同口径）：永远猜最常出现的那个是基线。

Run: .\.venv\Scripts\python.exe tests\test_attention_schema.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.attention_schema import AttentionSchema  # noqa: E402


def test_starts_silent() -> None:
    a = AttentionSchema()
    assert a.predict() == (None, 0.0)
    assert a.accuracy == 0.0 and a.bits_per_ignition == 0.0
    print("       no ignitions yet -> (None, 0.0)")


def test_learns_a_two_cycle() -> None:
    a = AttentionSchema()
    seq = [1, 2] * 30
    for u in seq:
        a.observe(u)
    assert a.accuracy > 0.85, a.accuracy
    assert a.accuracy > a.base_accuracy + 0.2, (a.accuracy, a.base_accuracy)
    assert a.bits_per_ignition < a.base_bits_per_ignition, (
        a.bits_per_ignition, a.base_bits_per_ignition)
    assert a.predict(1)[0] == 2 and a.predict(2)[0] == 1
    print(f"       two-cycle: acc {a.accuracy:.2f} vs base "
          f"{a.base_accuracy:.2f}; bits {a.bits_per_ignition:.3f} vs base "
          f"{a.base_bits_per_ignition:.3f}")


def test_prediction_confidence_is_bounded() -> None:
    a = AttentionSchema()
    for u in [3, 4, 3, 4, 3, 4]:
        a.observe(u)
    u, conf = a.predict(3)
    assert 0.0 < conf <= 1.0, conf
    assert a.mean_confidence > 0.0
    print(f"       predict(3)={u}, conf={conf:.3f}")


def test_roundtrip_keeps_learning() -> None:
    a = AttentionSchema()
    for u in [5, 6] * 8:
        a.observe(u)
    b = AttentionSchema.from_dict(a.to_dict())
    assert b.accuracy == a.accuracy and b.bits_per_ignition == a.bits_per_ignition
    b.observe(5)
    assert b.observations == a.observations + 1
    print(f"       roundtrip keeps counts; bits={b.bits_per_ignition:.3f}")


def test_new_symbol_is_not_scored_as_certain() -> None:
    """没见过的 uid 不能被 Laplace 算成概率 1（会给出 0 bits 的假好成绩）。"""
    a = AttentionSchema()
    a.observe(7)
    p = a._p(7, 99)
    assert p < 1.0, p
    print(f"       unseen uid p={p:.3f} (<1)")


def test_stream_wires_attention_without_changing_behaviour() -> None:
    """Stream 只在观测侧喂点火；存档带 attention；额外走 3 步不崩。"""
    from src.common import persist
    from src.L5_cognition.stream import Stream

    st = Stream()
    assert persist.load_l5(st), "需要存档"
    payload = persist.build_l5_payload(st)
    assert "attention" in payload, "payload 缺 attention"
    st2 = Stream()
    persist.apply_l5_payload(st2, payload)
    assert st2.attention.observations == st.attention.observations
    # 旧存档没有 attention 时，会用 workspace.history 预热；两边应一致，
    # 而且只要她点过火，样本数就不该是 0。
    if st.workspace.ignition_count:
        assert st2.attention.observations > 0, st2.attention.observations
    ss = st2.stream_state()
    for k in ("attn_n", "attn_acc", "attn_bits", "attn_base_bits", "attn_conf"):
        assert k in ss, k
    before = st2._step_count
    for _ in range(3):
        st2.landscape.set_noise_amplitude(0.02)
        st2.step()
    assert st2._step_count == before + 3
    assert st2.attention.observations >= st.attention.observations
    print(f"       attention wired: n={st2.attention.observations}, "
          f"bits={ss['attn_bits']}")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
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
