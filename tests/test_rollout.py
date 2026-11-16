r"""b 项：多步 rollout 的单元测试（不依赖真实存档）。

判据（docs/2026-11-10 §7 b 级）：误差随 k 增长但不爆，且优于同一批基线。

这里钉住三件与结论无关、但错了会让结论作废的事：
1. `rollout_distributions` 每一步的分布归一（V*C + Σnxt = 1）；
2. 一个手算得出来的小结构上，k=1、k=2 的概率精确对得上；
3. 打乱转移结构后，rollout 相对边际的增益消失（否则说明增益来自边际，不是结构）。

Run: .\.venv\Scripts\python.exe tests\test_rollout.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.rollout_audit import build, rollout_distributions  # noqa: E402


def _p(dists, k, t):
    C, nxt = dists[k]
    return C + nxt.get(t, 0.0)


def test_distributions_are_normalised() -> None:
    seq = [1, 2, 1, 2, 3, 1, 3, 3, 2, 1]
    counts, vocab, _ = build(seq)
    V = len(vocab)
    dists = rollout_distributions(counts, vocab, 1, 4, smoothing=0.5)
    for k, (C, nxt) in dists.items():
        total = V * C + sum(nxt.values())
        assert abs(total - 1.0) < 1e-9, (k, total)
    print(f"       V={V}: every k=1..4 sums to 1 within 1e-9")


def test_one_step_matches_hand_computation() -> None:
    """seq 里 1→2 出现 2 次、1→1 一次，于是 P^1(1)[2] 可以手算。"""
    seq = [1, 2, 1, 1, 2, 1]
    counts, vocab, _ = build(seq)
    V = len(vocab)
    alpha = 0.5
    dists = rollout_distributions(counts, vocab, 1, 2, smoothing=alpha)

    out = sum(counts[1].values())
    denom = out + alpha * V
    want_2 = (counts[1][2] + alpha) / denom
    want_1 = (counts[1][1] + alpha) / denom
    assert abs(_p(dists, 1, 2) - want_2) < 1e-9, (_p(dists, 1, 2), want_2)
    assert abs(_p(dists, 1, 1) - want_1) < 1e-9, (_p(dists, 1, 1), want_1)
    print(f"       k=1 hand-check ok: P(1->2)={want_2:.4f}, P(1->1)={want_1:.4f}")


def test_second_step_mixes_through_the_first() -> None:
    """P^2(1)[t] = Σ_s P^1(1)[s] · P^1(s)[t]，逐项对上。"""
    seq = [1, 2, 3, 1, 2, 3, 1, 2, 1]
    counts, vocab, _ = build(seq)
    alpha = 0.5
    dists = rollout_distributions(counts, vocab, 1, 2, smoothing=alpha)
    V = len(vocab)

    def step1(s: int) -> dict:
        row = counts.get(s)
        if not row:
            return {t: 1.0 / V for t in vocab}
        denom = sum(row.values()) + alpha * V
        return {t: (row.get(t, 0) + alpha) / denom for t in vocab}

    want = {t: 0.0 for t in vocab}
    for s, ps in step1(1).items():
        for t, pt in step1(s).items():
            want[t] += ps * pt
    for t in vocab:
        assert abs(_p(dists, 2, t) - want[t]) < 1e-9, (t, _p(dists, 2, t), want[t])
    print("       k=2 matches the two-step expansion over every region")


def test_a_flat_transition_has_no_more_information_than_marginal() -> None:
    """把转移结构打平（每个状态的出边都是同一个均匀分布），k 步就该退化成边际。

    这条是防「增益来自边际」的：结构没有信息时增益必须归零。
    """
    V = 5
    vocab = set(range(V))
    counts = {s: {t: 1 for t in range(V)} for s in range(V)}
    dists = rollout_distributions(counts, vocab, 0, 3, smoothing=1e-12)
    for k in (1, 2, 3):
        for t in range(V):
            p = _p(dists, k, t)
            assert abs(p - 1.0 / V) < 1e-6, (k, t, p)
    print("       flat structure: every k gives the uniform marginal, as it must")


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
