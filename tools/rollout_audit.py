r"""tools/rollout_audit.py — 审 b 项：她能不能做多步预测。

判据（docs/2026-11-10_能不能成功_可测边界.md §7，b 级）：
    同一批基线；误差随 k 增长但不爆。

为什么不用「贪心串接」
----------------------
一阶模型的贪心 k 步是 predict(当前) → predict(预测值) → ……。在真实数据上它几跳
之后就锁进一个循环，那时候量到的不是预测能力，是循环的不变量。真 rollout 是
**分布**：从当前区域出发，用她自己的转移表推 k 步之后的状态分布 P^k，再拿真实到达
的区域去打分。k 步分布可以精确算，不需要采样。

平滑怎么精确处理
----------------
每个状态的转移都加了 Laplace 平滑 sm，分母 denom_s = 出度_s + sm*V（V 是词表大小）。
于是对任意目标 t：

    t 在 s 的出边里 : P^1[s][t] = (c_st + sm) / denom_s
    t 不在          : P^1[s][t] = sm / denom_s

「不在」那半对每个 t 都一样，可以合并成一个常数：

    C_k = Σ_s p_s * sm / denom_s
    P^k[t] = C_k + Σ_{s→t} p_s * c_st / denom_s

状态 s 的 p_s 是上一轮的完整分布：P^{k-1}[s] = C_{k-1} + nxt_{k-1}[s]。
所以每步只需要遍历「有出边的状态」和当前 nxt 的键，不必碰 V×V。
没有出边的状态只贡献平滑质量，各 C/V，那一项用 (V - |keys|) 一次算掉。

基线（与 M1 审计同一批）
------------------------
    marginal  训练段的边际分布，k 步都用它（等价于「下一跳与历史无关」）
    same      0 阶：目标 = 起点区域，用训练段「同名频率」估

判据是「误差随 k 增长但不爆」：看 gain = marginal_bits − rollout_bits 随 k 的形状，
以及 k≥4 时是否仍为正。

用法：
    python tools\rollout_audit.py                  # 3000 步
    python tools\rollout_audit.py --steps 6000 --json

只读存档；噪声幅度固定（探针，不是线上环境）。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.self_model import SMOOTHING  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

KS = (1, 2, 4, 8)


def build(seq: list[int]) -> tuple[dict, set, set]:
    """一阶转移计数、词表、有出边的状态。"""
    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for a, b in zip(seq, seq[1:]):
        counts[int(a)][int(b)] += 1
    vocab = {int(x) for x in seq}
    return counts, vocab, set(counts)


def rollout_distributions(counts: dict, vocab: set, start: int, k_max: int,
                          smoothing: float = SMOOTHING) -> dict:
    """从 start 出发，k=1..k_max 每一步的完整分布。

    返回 {k: (C, nxt)}，对词表里任意 t：P^k[t] = C + nxt.get(t, 0.0)。
    """
    V = max(len(vocab), 1)
    active = set(counts)
    C = 0.0
    nxt: dict[int, float] = {int(start): 1.0}
    out: dict[int, tuple[float, dict]] = {}
    for step in range(1, k_max + 1):
        keys = active | set(nxt)
        C_new = float(V - len(keys)) * C / V if V > len(keys) else 0.0
        nn: dict[int, float] = {}
        for s in keys:
            p = nxt.get(s, 0.0) + C
            if p <= 0.0:
                continue
            row = counts.get(s)
            if not row:
                # 她从没离开过这个区域：平滑到全词表，各 p/V。
                C_new += p / V
                continue
            tot = float(sum(row.values()))
            denom = tot + smoothing * V
            C_new += p * smoothing / denom
            for t, c in row.items():
                nn[int(t)] = nn.get(int(t), 0.0) + p * float(c) / denom
        nxt = nn
        C = C_new
        out[step] = (C, dict(nn))
    return out


def evaluate(seq: list[int], ks=KS, points: int = 400,
             smoothing: float = SMOOTHING, seed: int = 0) -> dict:
    """前 60% 拟合，后 40% 打分。"""
    split = max(int(len(seq) * 0.6), 2)
    train, test = seq[:split], seq[split:]
    counts, vocab, _ = build(train)
    V = max(len(vocab), 1)
    marg = collections.Counter(int(x) for x in train)
    n_marg = float(sum(marg.values()))

    def p_marg(t: int) -> float:
        return (marg.get(int(t), 0.0) + smoothing) / (n_marg + smoothing * V)

    same_freq: dict[int, float] = {}
    for k in ks:
        same_freq[k] = sum(1 for a, b in zip(train, train[k:]) if a == b) / max(len(train) - k, 1)

    k_max = max(ks)
    hi = len(test) - k_max
    if hi <= 0:
        return {"error": "test segment too short"}
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(hi), min(points, hi)))

    top_marg = marg.most_common(1)[0][0]
    out: dict[int, dict] = {k: {"n": 0, "bits": {"rollout": 0.0, "marginal": 0.0, "same": 0.0},
                                "same_rate": 0.0, "top_marginal_rate": 0.0} for k in ks}
    for i in idx:
        start = int(test[i])
        # 一个起点只推一次，所有 k 共用这串分布。
        dists = rollout_distributions(counts, vocab, start, k_max, smoothing)
        for k in ks:
            target = int(test[i + k])
            C, nxt = dists[k]
            p = max(C + nxt.get(target, 0.0), 1e-12)
            d = out[k]
            d["bits"]["rollout"] += -math.log2(p)
            d["bits"]["marginal"] += -math.log2(p_marg(target))
            q = min(max(same_freq[k], 0.02), 0.98)
            p_same = q if target == start else (1.0 - q) / max(V - 1, 1)
            d["bits"]["same"] += -math.log2(max(p_same, 1e-12))
            d["same_rate"] += int(target == start)
            d["top_marginal_rate"] += int(target == top_marg)
            d["n"] += 1
    for k in ks:
        d = out[k]
        n = max(d["n"], 1)
        out[k] = {
            "n": d["n"],
            "bits": {name: round(v / n, 4) for name, v in d["bits"].items()},
            "gain_over_marginal": round((d["bits"]["marginal"] - d["bits"]["rollout"]) / n, 4),
            "same_rate": round(d["same_rate"] / n, 4),
            "top_marginal_rate": round(d["top_marginal_rate"] / n, 4),
        }
    return {
        "train": len(train),
        "test": len(test),
        "vocab": len(vocab),
        "by_k": out,
    }


def run(steps: int = 3000, amplitude: float = 0.02, seed: int = 0) -> dict:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive: run P2 first, or point state_dir at one")

    seq: list[int] = []
    for _ in range(steps):
        st.landscape.set_noise_amplitude(amplitude)
        st.step()
        aid = st.detector.current_attractor
        if aid is not None:
            seq.append(int(aid))

    return {
        "steps": int(steps),
        "amplitude": float(amplitude),
        "distinct_regions": len(set(seq)),
        **evaluate(seq, seed=seed),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="audit multi-step rollout (criterion b)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--amplitude", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    r = run(args.steps, args.amplitude, args.seed)
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    print(f"steps={r['steps']}  distinct regions={r['distinct_regions']}  "
          f"train={r['train']}  test={r['test']}  vocab={r['vocab']}")
    print(f"{'k':>3}{'rollout':>12}{'marginal':>12}{'same':>12}{'gain':>12}"
          f"{'P(same)':>10}{'P(top)':>10}")
    for k in sorted(r["by_k"]):
        d = r["by_k"][k]
        b = d["bits"]
        print(f"{k:>3}{b['rollout']:>12.4f}{b['marginal']:>12.4f}{b['same']:>12.4f}"
              f"{d['gain_over_marginal']:>12.4f}{d['same_rate']:>10.4f}"
              f"{d['top_marginal_rate']:>10.4f}")
    print("\ngain = marginal_bits - rollout_bits（正 = 她的多步分布比边际更准）")
    print("判据：gain 随 k 增长但不爆（不是指数崩掉），且 k>=4 时仍为正")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
