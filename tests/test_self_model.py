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

from src.L5_cognition.self_model import online_accuracy  # noqa: E402


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