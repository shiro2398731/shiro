r"""tools/departure_audit.py — b 项换靶：预测「k 步内会不会离开」，不是「落在哪」。

为什么要换
----------
`tools/rollout_audit.py` 的结论是：多步落点预测 k=1..8 全部输给「她还会待着」这条
0 阶规则（2.16 vs 6.59 bits）。原因不是转移表错了——她 79% 的步原地不动，任何预测
「下一跳去哪」的模型都要和「不动」抢分数，而后者一个参数就够。

但「她还会待着」这条规则本身对「**什么时候**离开」没有信息。b 级的判据原本就是
「误差随 k 增长但不爆」：落点会把这件事盖住，离开时间不会。

判据
----
从 t 时刻出发，预测「接下来 k 步内是否离开当前区域」，用对数损失比：

    her      P(离开) = 1 − P^k(当前)[当前]，P^k 来自她自己的转移表（多步 rollout）
    constant 训练段的全局离开率
    survival 训练段停留时长均值的指数生存：P(离开) = 1 − exp(−k/mean_dwell)

`her` 赢过两条基线，才说明她的多步结构里有「离开时间」这件事。

用法：
    python tools\departure_audit.py
    python tools\departure_audit.py --steps 3000 --json

只读存档。
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
from tools.rollout_audit import build, rollout_distributions  # noqa: E402

KS = (1, 2, 4, 8)


def _dwell_lengths(seq: list[int]) -> list[int]:
    """训练段里每一段连续相同区域的长度。"""
    runs: list[int] = []
    if not seq:
        return runs
    cur, n = seq[0], 1
    for x in seq[1:]:
        if x == cur:
            n += 1
        else:
            runs.append(n)
            cur, n = x, 1
    runs.append(n)
    return runs


def _logloss(y: int, p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(math.log2(p) if y else math.log2(1.0 - p))


def evaluate(seq: list[int], ks=KS, points: int = 500,
             smoothing: float = SMOOTHING, seed: int = 0) -> dict:
    """前向（prequential）评估。

    2026-11-16：一开始用的是「前 60% 拟合、后 40% 打分」，结果测试段 1200 个起点里
    有 1143 个是训练段没见过的区域——她的世界一直在长新区域，按时间切分必然右偏。
    那些点只能预测「没见过 ⇒ 均匀」，于是 P(离开) 被算成 0.99，而真实是 0.21，
    分数看着烂，烂的是切分方式。

    她真实的处境是一直在学：预测 t+1 时，t 之前的数据都已经看过。所以这里每向前走
    一步，就把刚走过的转移并进表里，再预测接下来的 k 步。
    """
    k_max = max(ks)
    warmup = max(int(len(seq) * 0.3), 50)
    if len(seq) - warmup <= k_max:
        return {"error": "sequence too short"}

    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    vocab: set[int] = set(seq[:warmup])
    for a, b in zip(seq[:warmup], seq[1:warmup]):
        counts[int(a)][int(b)] += 1

    rng = random.Random(seed)
    lo, hi = warmup, len(seq) - k_max
    idx = sorted(rng.sample(range(lo, hi), min(points, hi - lo)))

    runs = _dwell_lengths(seq[:warmup])
    mean_dwell = float(sum(runs)) / max(len(runs), 1)
    left = sum(1 for a, b in zip(seq[:warmup], seq[1:warmup]) if a != b)
    p_leave_const = left / max(warmup - 1, 1)

    out: dict[int, dict] = {}
    cursor = warmup
    for k in ks:
        out[k] = {"n": 0, "leave_rate": 0.0, "bits": {"her": 0.0, "constant": 0.0, "survival": 0.0}}
        out[k]["p_surv"] = 1.0 - math.exp(-k / max(mean_dwell, 1e-9))

    for i in idx:
        # 把 cursor..i 之间走过的每一步并进表里：这就是「她一直在学」。
        for j in range(cursor, i):
            counts[int(seq[j])][int(seq[j + 1])] += 1
            vocab.add(int(seq[j]))
            vocab.add(int(seq[j + 1]))
        cursor = max(cursor, i)

        start = int(seq[i])
        V = max(len(vocab), 1)
        # 全局离开率也随着她走的部分更新
        n_obs = max(i - warmup, 1)
        left_far = sum(1 for a, b in zip(seq[warmup:i], seq[warmup + 1:i + 1]) if a != b)
        p_const = (left + left_far) / max((warmup - 1) + n_obs, 1)
        p_const = min(max(p_const, 1e-6), 1.0 - 1e-6)

        dists = rollout_distributions(counts, vocab, start, k_max, smoothing)
        for k in ks:
            y = int(any(int(seq[i + j]) != start for j in range(1, k + 1)))
            C, nxt = dists[k]
            p_stay = C + nxt.get(start, 0.0)
            p_her = min(max(1.0 - p_stay, 1e-9), 1.0 - 1e-9)
            ps = min(max(out[k]["p_surv"], 1e-6), 1.0 - 1e-6)
            d = out[k]
            d["bits"]["her"] += _logloss(y, p_her)
            d["bits"]["constant"] += _logloss(y, p_const)
            d["bits"]["survival"] += _logloss(y, ps)
            d["leave_rate"] += y
            d["n"] += 1

    for k in ks:
        d = out[k]
        n = max(d["n"], 1)
        out[k] = {
            "n": d["n"],
            "leave_rate": round(d["leave_rate"] / n, 4),
            "bits": {name: round(v / n, 4) for name, v in d["bits"].items()},
            "gain_vs_constant": round((d["bits"]["constant"] - d["bits"]["her"]) / n, 4),
            "gain_vs_survival": round((d["bits"]["survival"] - d["bits"]["her"]) / n, 4),
        }
    return {
        "mode": "prequential",
        "warmup": warmup,
        "points": len(idx),
        "vocab": len(vocab),
        "mean_dwell_warmup": round(mean_dwell, 2),
        "p_leave_warmup": round(p_leave_const, 4),
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
    ap = argparse.ArgumentParser(description="b: predict departure, not destination")
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
    print(f"mode={r['mode']}  mean dwell (warmup)={r['mean_dwell_warmup']}  "
          f"leave rate (warmup)={r['p_leave_warmup']}")
    print(f"{'k':>3}{'leave':>8}{'her':>10}{'constant':>10}{'survival':>10}"
          f"{'vs const':>10}{'vs surv':>10}")
    for k in sorted(r["by_k"]):
        d = r["by_k"][k]
        b = d["bits"]
        print(f"{k:>3}{d['leave_rate']:>8.4f}{b['her']:>10.4f}{b['constant']:>10.4f}"
              f"{b['survival']:>10.4f}{d['gain_vs_constant']:>10.4f}"
              f"{d['gain_vs_survival']:>10.4f}")
    print("\n正增益 = 她的多步分布比基线更会判断「什么时候离开」")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
