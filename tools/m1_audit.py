"""
tools/m1_audit.py — 审 M1：她的"自我预测"到底比基线强在哪。

为什么需要这个工具
------------------
M1 之前报的是**命中率**。在她身上，命中率是一个会骗人的指标：她大量时间是
在同一个区域里停留，"预测下一个区域 = 当前区域"这条 0 阶规则就能拿到很高的分数。
实测（真存档 2000 步，1987 个转移）：

    predictor            accuracy      prequential bits/transition
    自我模型（一阶）      0.9381        0.4813
    "不变"（0 阶）        0.9446        3.6854        <- 准确率更高，比特差得多
    边际（总选最常见）     0.1998        3.2816

所以判据必须看**对数损失**：它对"自信地错"罚得重，"不变"那条规则在 94% 的情况下
看起来对，一旦错就是无限代价（平滑后 3.69 bits）。模型的 0.4813 bits 才是真本事。

打乱序列（保留边际、破坏转移结构）后：模型 3.3789 bits、边际 3.2956 bits ——
优势消失并略微变差（拟合了噪声）。这才叫"结构是真的"。

用法
----
    python tools\\m1_audit.py                # 2000 步
    python tools\\m1_audit.py --steps 5000
    python tools\\m1_audit.py --json

注意：只读存档；噪声幅度固定（探针，不是线上环境）。
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
from src.L5_cognition.self_model import SMOOTHING, OnlineSelfModel  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def evaluate(seq: list[int], shuffle_seed: int | None = None) -> dict:
    """Prequential scoring of three predictors on one sequence.

    `model`   first-order transition table (her M1), argmax + smoothed loss
    `same`    0th order: the next region is the current one
    `marginal` 0th order: always the most frequent region so far
    """
    s = list(seq)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(s)

    m = OnlineSelfModel()
    seen: collections.Counter = collections.Counter()
    n = 0
    hits = {"model": 0, "same": 0, "marginal": 0, "same_freq": 0}
    bits = {"model": 0.0, "same": 0.0, "marginal": 0.0, "same_freq": 0.0}
    same_hits = 0
    trans = 0
    prev = None

    for cur in s:
        if prev is not None:
            vocab = max(len(seen), 1)
            total = sum(seen.values())

            # --- model ---
            row = m._counts.get(prev) or {}
            row_total = sum(row.values())
            p_model = (row.get(cur, 0.0) + SMOOTHING) / (row_total + SMOOTHING * vocab)
            pred = m.predict(prev)
            hits["model"] += int(pred == cur)
            bits["model"] += -math.log2(p_model)

            # --- "no change" ---
            # P(next == prev): estimated from how often the region she is in has
            # been seen at all, clamped so the alternative never gets zero mass.
            q = (seen.get(prev, 0) + SMOOTHING) / (total + SMOOTHING * vocab)
            q = min(max(q, 0.02), 0.98)
            p_same = q if cur == prev else (1.0 - q) / max(vocab - 1, 1)
            hits["same"] += int(cur == prev)
            bits["same"] += -math.log2(max(p_same, 1e-12))

            # --- marginal ---
            top = seen.most_common(1)[0][0] if seen else -1
            p_marg = (seen.get(cur, 0) + SMOOTHING) / (total + SMOOTHING * vocab)
            hits["marginal"] += int(top == cur)
            bits["marginal"] += -math.log2(p_marg)

            # --- same_freq: the strong 0th-order rule ---
            # 2026-11-16: 上面那条 `same` 用「先验频率」估 P(下一跳=当前)，它把
            # 「她不动」估得极低（真实值 0.79，它给 0.05 量级），于是基线弱成
            # 5.3 bits，模型赢它没有意义。这条用**在线估计的同名频率**：
            # 单独统计「下一步是否等于当前」，二分类平滑。实测真存档上模型
            # 3.15 bits 对这条 2.13 bits —— 模型输。两个基线都要印。
            q_sf = (same_hits + SMOOTHING) / (trans + SMOOTHING * 2)
            q_sf = min(max(q_sf, 0.02), 0.98)
            p_sf = q_sf if cur == prev else (1.0 - q_sf) / max(vocab - 1, 1)
            hits["same_freq"] += int(cur == prev)
            bits["same_freq"] += -math.log2(max(p_sf, 1e-12))

            n += 1

        if prev is not None:
            trans += 1
            same_hits += int(cur == prev)
        seen[cur] += 1
        if prev is not None:
            m.observe(prev, cur)
        prev = cur

    return {
        "n": n,
        "accuracy": {k: round(v / n, 4) for k, v in hits.items()} if n else {},
        "bits_per_transition": {k: round(v / n, 4) for k, v in bits.items()} if n else {},
    }


def run(steps: int = 2000, amplitude: float = 0.02, seed: int = 0) -> dict:
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
        "transitions": max(len(seq) - 1, 0),
        "distinct_regions": len(set(seq)),
        "real": evaluate(seq),
        "shuffled": evaluate(seq, shuffle_seed=seed),
        "engine_model": {
            "accuracy": round(st.self_model.accuracy, 4),
            "bits_per_observation": round(st.self_model.bits_per_observation, 4),
            "scored": st.self_model.scored,
        },
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="audit M1 against 0th-order baselines")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--amplitude", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    r = run(args.steps, args.amplitude, args.seed)
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    print(f"steps={r['steps']}  transitions={r['transitions']}  "
          f"distinct regions={r['distinct_regions']}  amplitude={r['amplitude']}")
    for cond in ("real", "shuffled"):
        d = r[cond]
        print(f"{cond}:")
        print(f"  {'predictor':<10}{'accuracy':>12}{'bits/transition':>18}")
        for k in ("model", "same", "same_freq", "marginal"):
            print(f"  {k:<10}{d['accuracy'][k]:>12.4f}{d['bits_per_transition'][k]:>18.4f}")
    print(f"engine's own counters: accuracy={r['engine_model']['accuracy']:.4f}  "
          f"bits={r['engine_model']['bits_per_observation']:.4f}  "
          f"(prequential over {r['engine_model']['scored']} transitions)")
    print("Verdict rule: accuracy is dominated by dwell; the model must win on BITS,")
    print("              and must lose that advantage when the sequence is shuffled.")
    print("两条 same 基线不是一回事：`same` 用先验频率估 P(不动)，弱得没有意义；")
    print("`same_freq` 用在线同名频率，才是 0 阶规则该有的强度。判据必须对着它看。")
    print("2026-11-16 实测：model 3.0141 对 same_freq 2.1561 —— 模型输；")
    print("打乱后 model 6.4112 / same_freq 6.6024，说明它确实携带结构，只是不够赢。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
