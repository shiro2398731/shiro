r"""M2：她的把握与她的命中率是否一致（校准）。

M1 问"她猜得准不准"；M2 问"她知不知道自己准不准"。这两件事可以分开失败，
所以各自要有判据，而且 M2 的判据单独看会骗人：

    **校准好 ≠ 知道。** 一个永远输出 1/chance 的模型也是完美校准的。

因此这里的测试分四种情形，缺一种就分不清"她会评估自己"与"她恒等于运气"：

1. 结构清楚时：把握高、命中高、ECE 小；
2. 纯噪声时：命中接近 chance、ECE 也小 —— 校准好但什么都没学到（必须一起断言）；
3. 学会一段、拿去预测另一段（冻结）：把握与命中脱节，ECE 必须变大；
4. `distribution()` 与 `predict()` 说的是同一个模型（argmax 一致、概率和 ≤ 1）。

实测背景（她的存档，1200 步、31 个区域，`tools/m2_audit.py`）：
    real        acc 0.8964 / conf 0.6994 / ECE 0.1987
    shuffled    acc 0.0685 / conf 0.1091 / ECE 0.0406
    transferred acc 0.8692 / conf 0.4583 / ECE 0.4280
三个条件都指向同一件事：**她低估自己**（低把握那一档 conf 0.356 / acc 0.906）。

Run: .\.venv\Scripts\python.exe tests\test_m2.py
"""

from __future__ import annotations

import os
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

from src.L5_cognition.self_model import OnlineSelfModel  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tools.m2_audit import calibration  # noqa: E402


def _cycle(n: int, states: int = 4, offset: int = 0) -> list[int]:
    return [(i + offset) % states for i in range(n)]


def _score(model: OnlineSelfModel, seq: list[int]) -> list[tuple[float, bool]]:
    rows = []
    prev = None
    for cur in seq:
        if prev is not None:
            pred = model.predict(prev)
            if pred is not None:
                rows.append((model.confidence(prev), bool(pred == cur)))
            model.observe(prev, cur)
        prev = cur
    return rows


def test_confidence_matches_the_choice() -> None:
    m = OnlineSelfModel()
    for a, b in [(0, 1), (1, 2), (2, 0), (0, 1), (1, 2)]:
        m.observe(a, b)
    for s in (0, 1, 2):
        dist = m.distribution(s)
        assert dist, s
        assert abs(sum(dist.values()) - 1.0) < 1e-9 or sum(dist.values()) <= 1.0, dist
        best = max(dist.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        assert m.predict(s) == best, (s, m.predict(s), best)
        assert abs(m.confidence(s) - dist[m.predict(s)]) < 1e-12
    print("       argmax(distribution) == predict(); confidence == mass on that choice")


def test_structured_sequence_is_calibrated_and_accurate() -> None:
    rows = _score(OnlineSelfModel(), _cycle(800))
    d = calibration(rows)
    assert d["accuracy"] > 0.9, d
    assert d["mean_conf"] > 0.5, d
    assert d["ece"] < 0.2, d
    print(f"       cycle: acc={d['accuracy']:.3f} conf={d['mean_conf']:.3f} "
          f"ECE={d['ece']:.3f}")


def test_calibration_alone_is_not_knowledge() -> None:
    """Noise: calibrated, and still ignorant. The test asserts both halves."""
    rnd = random.Random(5)
    seq = [rnd.randrange(5) for _ in range(2000)]
    d = calibration(_score(OnlineSelfModel(), seq))
    chance = 1.0 / 5.0
    assert abs(d["accuracy"] - chance) < 0.1, d
    assert d["ece"] < 0.1, d
    assert d["mean_conf"] < 2.0 * chance, d
    print(f"       noise: acc={d['accuracy']:.3f} (chance {chance:.2f}) "
          f"conf={d['mean_conf']:.3f} ECE={d['ece']:.3f} -- calibrated but ignorant")


def test_confident_and_wrong_shows_up_as_calibration_error() -> None:
    """Learn one cycle, then predict a different one with the table frozen."""
    trained = OnlineSelfModel()
    train = _cycle(600)
    for a, b in zip(train, train[1:]):
        trained.observe(a, b)

    test = [(-i) % 4 for i in range(400)]   # next symbol goes the other way: the learned rule is wrong
    rows = []
    for a, b in zip(test, test[1:]):
        rows.append((trained.confidence(a), bool(trained.predict(a) == b)))
    d = calibration(rows)
    assert d["ece"] > 0.15, d
    assert d["accuracy"] < d["mean_conf"], d
    print(f"       transfer: acc={d['accuracy']:.3f} conf={d['mean_conf']:.3f} "
          f"ECE={d['ece']:.3f} (over-confident, detected)")


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
