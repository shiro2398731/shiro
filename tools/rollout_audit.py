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
    """前向（prequential）评估，两个视图。

    2026-11-16：和 `departure_audit` 一样，一开始用的是「前 60% 拟合 / 后 40% 打分」。
    她的世界一直在长新区域，按时间切分时测试段 95% 的起点是训练段没见过的，
    分数失真。改成前向：每走一步就把刚走过的转移并进表里，再预测接下来的 k 步。

    两个视图：
      all    所有起点，预测 k 步后落在哪
      moved  只算「k 步后确实离开了起点」的样本，预测落点。
             79% 的步原地不动，落点问题的分数一直被这部分盖住；给定离开之后，
             那才是转移表该赢的地方。
    """
    k_max = max(ks)
    warmup = max(int(len(seq) * 0.3), 50)
    if len(seq) - warmup <= k_max:
        return {"error": "sequence too short"}

    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    vocab: set[int] = set(seq[:warmup])
    marg: collections.Counter = collections.Counter()
    for a, b in zip(seq[:warmup], seq[1:warmup + 1]):
        counts[int(a)][int(b)] += 1
        marg[int(b)] += 1

    rng = random.Random(seed)
    lo, hi = warmup, len(seq) - k_max
    idx = sorted(rng.sample(range(lo, hi), min(points, hi - lo)))

    acc: dict[int, dict] = {
        k: {"n": 0, "n_moved": 0,
            "bits_all": 0.0, "bits_moved": 0.0, "bits_moved_marg": 0.0,
            "same_rate": 0} for k in ks
    }
    cursor = warmup
    for i in idx:
        for j in range(cursor, i):
            counts[int(seq[j])][int(seq[j + 1])] += 1
            vocab.add(int(seq[j]))
            vocab.add(int(seq[j + 1]))
            marg[int(seq[j + 1])] += 1
        cursor = max(cursor, i)

        start = int(seq[i])
        V = max(len(vocab), 1)
        n_marg = float(sum(marg.values())) or 1.0
        dists = rollout_distributions(counts, vocab, start, k_max, smoothing)
        for k in ks:
            target = int(seq[i + k])
            C, nxt = dists[k]
            p = C + nxt.get(target, 0.0)
            d = acc[k]
            d["bits_all"] += -math.log2(max(p, 1e-12))
            d["same_rate"] += int(target == start)
            if target != start:
                # 给定离开：条件分布 = 联合 / P(离开)
                stay = C + nxt.get(start, 0.0)
                p_leave = max(1.0 - stay, 1e-9)
                p_cond = p / p_leave
                p_m = (marg.get(target, 0) + smoothing) / (n_marg + smoothing * V)
                d["bits_moved"] += -math.log2(max(p_cond, 1e-12))
                d["bits_moved_marg"] += -math.log2(max(p_m, 1e-12))
                d["n_moved"] += 1
            d["n"] += 1

    out: dict[int, dict] = {}
    for k in ks:
        d = acc[k]
        n = max(d["n"], 1)
        nm = max(d["n_moved"], 1)
        out[k] = {
            "n": d["n"],
            "n_moved": d["n_moved"],
            "same_rate": round(d["same_rate"] / n, 4),
            "bits_all": round(d["bits_all"] / n, 4),
            "bits_moved": round(d["bits_moved"] / nm, 4),
            "bits_moved_marg": round(d["bits_moved_marg"] / nm, 4),
            "gain_moved": round((d["bits_moved_marg"] - d["bits_moved"]) / nm, 4),
        }
    return {
        "mode": "prequential",
        "warmup": warmup,
        "points": len(idx),
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

    print(f"steps={r['steps']}  regions={r['distinct_regions']}  "
          f"warmup={r['warmup']}  points={r['points']}  vocab={r['vocab']}")
    print(f"{'k':>3}{'all':>10}{'moved':>10}{'marg':>10}{'gain':>10}{'n_moved':>9}{'P(same)':>9}")
    for k in sorted(r["by_k"]):
        d = r["by_k"][k]
        print(f"{k:>3}{d['bits_all']:>10.4f}{d['bits_moved']:>10.4f}{d['bits_moved_marg']:>10.4f}"
              f"{d['gain_moved']:>10.4f}{d['n_moved']:>9}{d['same_rate']:>9.4f}")
    print("\nall   = 所有起点，预测落点（会被「她还在原地」盖住）")
    print("moved = 只算确实离开了的样本，预测落点（转移表该赢的地方）")
    print("gain  = marginal_bits - rollout_bits（正 = 比边际更准）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
