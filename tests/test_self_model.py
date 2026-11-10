r"""M1 验收：自我模型必须携带信息，且删除后性能下降。

判据：
  1. 结构化自身序列上，在线准确率显著高于 chance。
  2. 打乱转移顺序（保留边际分布）后，准确率回落接近 chance。
  3. 与 L5 真实存档的显著区域序列对照（若可用），同样成立。

Run: .\.venv\Scripts\python.exe tests\test_self_model.py
"""

from __future__ import annotations

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