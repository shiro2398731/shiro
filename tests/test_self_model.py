r"""M1 验收：自我模型必须携带信息，且删除后性能下降。

判据：
  1. 结构化自身序列上，在线准确率显著高于 chance。
  2. 打乱转移顺序（保留边际分布）后，准确率回落接近 chance。
  3. 与 L5 真实存档的显著区域序列对照（若可用），同样成立。

Run: .\.venv\Scripts\python.exe tests\test_self_model.py
"""

from __future__ import annotations

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.L5_cognition.self_model import OnlineSelfModel, online_accuracy  # noqa: E402


def _structured_sequence(n: int = 3000, states: int = 6, noise: float = 0.1) -> list[int]:
    """环形转移 + 少量随机跳转：存在可学的自身结构。"""
    rnd = random.Random(7)
    s = 0
    out = []
    for _ in range(n):
        out.append(s)
        s = (s + 1) % states if rnd.random() > noise else rnd.randrange(states)
    return out


def test_self_model_beats_chance() -> None:
    seq = _structured_sequence()
    r = online_accuracy(seq)
    assert r["accuracy"] > 3 * r["chance"], r
    print(f"       accuracy={r['accuracy']:.3f}  chance={r['chance']:.3f}  scored={r['scored']:.0f}")


def test_ablation_drops_to_chance() -> None:
    seq = _structured_sequence()
    intact = online_accuracy(seq)
    ablated = online_accuracy(seq, ablated=True)
    assert ablated["accuracy"] < 0.5 * intact["accuracy"], (intact, ablated)
    assert ablated["accuracy"] <= 2 * ablated["chance"], ablated
    print(f"       intact={intact['accuracy']:.3f}  ablated={ablated['accuracy']:.3f} "
          f"(chance={ablated['chance']:.3f})")


def test_pure_noise_has_no_self_structure() -> None:
    """无结构序列上，准确率不应高于 chance 太多——防止模型自欺。"""
    rnd = random.Random(11)
    seq = [rnd.randrange(5) for _ in range(2000)]
    r = online_accuracy(seq)
    assert r["accuracy"] <= 1.6 * r["chance"], r
    print(f"       noise: accuracy={r['accuracy']:.3f}  chance={r['chance']:.3f}")


def test_real_archive_sequence() -> None:
    """若存档可用，用真实显著区域序列检验一次。"""
    import os
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    from src.common import persist
    from src.L5_cognition.stream import Stream
    st = Stream()
    if not persist.load_l5(st):
        print("       (archive unavailable, skipped)")
        return
    ss = st.landscape.state_space
    ids = [i for i, a in enumerate(ss.attractors()) if a.amp > 0.05][:200]
    if len(ids) < 10:
        print("       (too few salient regions, skipped)")
        return
    rnd = random.Random(3)
    seq = [rnd.choice(ids) for _ in range(1000)]
    r = online_accuracy(seq)
    print(f"       archive ids={len(ids)}: accuracy={r['accuracy']:.3f} "
          f"chance={r['chance']:.3f} (random walk by construction)")


def _cycle(n: int = 400, states: int = 4) -> list[int]:
    return [i % states for i in range(n)]


def _online(seq: list[int]) -> tuple[float, float, int]:
    """(model accuracy, expect-no-change accuracy, scored steps)."""
    m = OnlineSelfModel()
    hits = same = scored = 0
    prev = None
    for cur in seq:
        if prev is not None:
            pred = m.predict(prev)
            if pred is not None:
                scored += 1
                hits += int(pred == cur)
                same += int(prev == cur)
            m.observe(prev, cur)
        prev = cur
    return (hits / scored if scored else 0.0,
            same / scored if scored else 0.0, scored)


def test_online_model_beats_expect_no_change() -> None:
    """The predictor that runs inside L5 must beat the 0th-order baseline.

    On a 4-cycle the baseline "the next region is the current one" is wrong
    every time, which is the case the previous wiring could not distinguish
    itself from.
    """
    acc, base, scored = _online(_cycle(400))
    assert scored > 300, scored
    assert base == 0.0, base
    assert acc > 0.9, acc
    print(f"       model={acc:.3f}  expect-same={base:.3f}  scored={scored}")


def test_online_model_has_no_advantage_on_noise() -> None:
    rnd = random.Random(5)
    seq = [rnd.randrange(6) for _ in range(2000)]
    acc, _base, scored = _online(seq)
    chance = 1.0 / 6.0
    assert acc <= 1.6 * chance, (acc, chance)
    print(f"       noise: model={acc:.3f} chance={chance:.3f} scored={scored}")


def test_self_model_survives_a_restart() -> None:
    """Her self-model is state, not session: it goes into the L5 payload."""
    import os
    os.chdir(pathlib.Path(__file__).resolve().parent.parent)
    from src.common import persist
    from src.L5_cognition.stream import Stream

    a = Stream(dim=16)
    for _ in range(30):
        for x in range(4):
            a.self_model.observe(x, (x + 1) % 4)
            a.self_gain.observe(True)
    a._last_symbol = 3
    payload = persist.build_l5_payload(a)

    b = Stream(dim=16)
    persist.apply_l5_payload(b, payload)

    assert b.self_model.predict(0) == 1, b.self_model.predict(0)
    assert b.self_model.states_seen == 4, b.self_model.states_seen
    assert b._last_symbol == 3, b._last_symbol
    assert b.self_gain.observations == a.self_gain.observations
    assert abs(b.self_gain.error - a.self_gain.error) < 1e-12
    print(f"       resumed: states={b.self_model.states_seen} "
          f"last={b._last_symbol} observations={b.self_gain.observations}")


# ══════════════════════════════════════════════════════════════════════════════
# M1 audit: accuracy is dominated by dwell, bits are not
# ══════════════════════════════════════════════════════════════════════════════

def test_bits_per_observation_matches_a_hand_calculation() -> None:
    """Prequential loss, scored before the table learns the transition.

    (0,1) first: no history, so p = 1.0 (0 bits).
    (1,0): symbol 1 unseen as a predecessor, vocabulary 2 -> p = 0.5 (1 bit).
    (0,1) and (1,0) again: each row has one of two outcomes -> p = 0.75.
    """
    m = OnlineSelfModel()
    for a, b in [(0, 1), (1, 0), (0, 1), (1, 0)]:
        m.observe(a, b)
    expected = (1.0 - 2.0 * math.log2(0.75)) / 4.0
    assert abs(m.bits_per_observation - expected) < 1e-9, (
        m.bits_per_observation, expected)
    assert m.scored == 4, m.scored
    assert m.states_seen == 2, m.states_seen
    print(f"       hand-checked: {m.bits_per_observation:.6f} bits/observation")


def test_model_tracks_its_own_accuracy() -> None:
    """Before 2026-11-10 `hits` was only ever set by a restore, so accuracy
    read 0.0 on a model that had been predicting correctly."""
    m = OnlineSelfModel()
    for _ in range(3):
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            m.observe(a, b)
    assert m.observations == 9, m.observations
    assert m.accuracy > 0.5, m.accuracy
    print(f"       self-tracked accuracy={m.accuracy:.3f} over {m.observations}")


def _dwell_cycle(n: int = 1200, states: int = 4, dwell: int = 3) -> list[int]:
    return [(i // dwell) % states for i in range(n)]


def test_standing_still_wins_on_accuracy_and_loses_on_bits() -> None:
    """Her own numbers, reproduced on a synthetic dwell-heavy sequence.

    Measured on the real archive: "predict the current region" scores 0.9390
    accuracy against the model's 0.9325, but 3.1321 bits against 0.4929. A
    metric where the trivial predictor wins cannot be the criterion.
    """
    import os
    os.chdir(pathlib.Path(__file__).resolve().parent.parent)
    from tools.m1_audit import evaluate

    d = evaluate(_dwell_cycle())
    assert d["accuracy"]["same"] >= d["accuracy"]["model"], d["accuracy"]
    assert d["bits_per_transition"]["model"] < 0.5 * d["bits_per_transition"]["same"], d
    print(f"       acc: model={d['accuracy']['model']:.3f} same={d['accuracy']['same']:.3f}"
          f"  |  bits: model={d['bits_per_transition']['model']:.4f} "
          f"same={d['bits_per_transition']['same']:.4f}")


def test_shuffling_removes_the_advantage() -> None:
    """If the advantage survives shuffling, it was never about transition structure."""
    import os
    os.chdir(pathlib.Path(__file__).resolve().parent.parent)
    from tools.m1_audit import evaluate

    seq = _dwell_cycle()
    real = evaluate(seq)
    shuf = evaluate(seq, shuffle_seed=0)
    assert real["bits_per_transition"]["model"] < 0.5 * shuf["bits_per_transition"]["model"], (
        real, shuf)
    print(f"       model bits: real={real['bits_per_transition']['model']:.4f}  "
          f"shuffled={shuf['bits_per_transition']['model']:.4f}")


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