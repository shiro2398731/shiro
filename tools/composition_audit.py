r"""tools/composition_audit.py — d 项：组合泛化。

判据（docs/2026-11-10 §7，d 级）：
    见过 A、B 之后能处理 A∘B（留出组合）。

这里把「留出组合」落成一个能算的东西：**没见过的区域对**。

对每个预测点，看 (上一个区域, 当前区域) 这一对在**它之前**的历史里出现过几次：

    seen    这一对出现过 -> 转移表有货
    unseen  这一对从没出现过 -> 她是"第一次走这条路"

判据：在 `unseen` 那部分上，她的转移表还能不能优于边际基线。
优于 -> 存在组合泛化（能从 A、B 各自的样子推出 A∘B）。
持平或更差 -> 只是查表，没见过的就得靠猜。

前向（prequential）评估，理由同 departure_audit：她的世界一直在长新区域，
按时间切分会让测试段 95% 是没见过的起点。

用法：
    python tools\composition_audit.py
    python tools\composition_audit.py --steps 4000 --json

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


def evaluate(seq: list[int], points: int = 600, smoothing: float = 0.02,
             seed: int = 0) -> dict:
    warmup = max(int(len(seq) * 0.3), 50)
    if len(seq) - warmup < 2:
        return {"error": "sequence too short"}

    counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    marg: collections.Counter = collections.Counter()
    vocab: set[int] = set(seq[:warmup])
    for a, b in zip(seq[:warmup], seq[1:warmup + 1]):
        counts[int(a)][int(b)] += 1
        marg[int(b)] += 1

    rng = random.Random(seed)
    lo, hi = warmup, len(seq) - 1
    idx = sorted(rng.sample(range(lo, hi), min(points, hi - lo)))

    acc = {
        "seen": {"n": 0, "bits_model": 0.0, "bits_marg": 0.0, "hits": 0, "top_hits": 0},
        "unseen": {"n": 0, "bits_model": 0.0, "bits_marg": 0.0, "hits": 0, "top_hits": 0},
    }
    cursor = warmup
    for i in idx:
        for j in range(cursor, i):
            counts[int(seq[j])][int(seq[j + 1])] += 1
            marg[int(seq[j + 1])] += 1
            vocab.add(int(seq[j]))
            vocab.add(int(seq[j + 1]))
        cursor = max(cursor, i)

        prev, cur = int(seq[i]), int(seq[i + 1])
        V = max(len(vocab), 1)
        n_marg = float(sum(marg.values())) or 1.0
        row = counts.get(prev) or {}
        tot = float(sum(row.values()))
        # 「这一对见过」= 这个具体目标在从这个起点出发的历史里出现过
        seen = int(row.get(cur, 0)) > 0
        bucket = acc["seen" if seen else "unseen"]

        p_model = (row.get(cur, 0.0) + smoothing) / (tot + smoothing * V)
        p_marg = (marg.get(cur, 0) + smoothing) / (n_marg + smoothing * V)
        bucket["bits_model"] += -math.log2(max(p_model, 1e-12))
        bucket["bits_marg"] += -math.log2(max(p_marg, 1e-12))
        top = marg.most_common(1)[0][0] if marg else -1
        pred = max(row.items(), key=lambda kv: kv[1])[0] if row else None
        bucket["hits"] += int(pred == cur)
        bucket["top_hits"] += int(top == cur)
        bucket["n"] += 1

    out = {}
    for name, d in acc.items():
        n = max(d["n"], 1)
        out[name] = {
            "n": d["n"],
            "bits_model": round(d["bits_model"] / n, 4),
            "bits_marg": round(d["bits_marg"] / n, 4),
            "gain": round((d["bits_marg"] - d["bits_model"]) / n, 4),
            "hit_rate": round(d["hits"] / n, 4),
            "top_marginal_rate": round(d["top_hits"] / n, 4),
        }
    total = acc["seen"]["n"] + acc["unseen"]["n"]
    out["share_unseen"] = round(acc["unseen"]["n"] / max(total, 1), 4)
    return out


def run(steps: int = 4000, amplitude: float = 0.02, seed: int = 0) -> dict:
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
    r = evaluate(seq, seed=seed)
    r.update({"steps": int(steps), "distinct_regions": len(set(seq))})
    return r


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="d: composition generalisation")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--amplitude", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    r = run(args.steps, args.amplitude, args.seed)
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    print(f"steps={r['steps']}  regions={r['distinct_regions']}  "
          f"unseen-pair share={r['share_unseen']:.3f}")
    print(f"{'bucket':<8}{'n':>7}{'model':>10}{'marginal':>10}{'gain':>10}"
          f"{'hit':>8}{'top':>8}")
    for name in ("seen", "unseen"):
        d = r[name]
        print(f"{name:<8}{d['n']:>7}{d['bits_model']:>10.4f}{d['bits_marg']:>10.4f}"
              f"{d['gain']:>10.4f}{d['hit_rate']:>8.4f}{d['top_marginal_rate']:>8.4f}")
    print("\nseen   = 这一步走的区域对，之前见过")
    print("unseen = 之前从没走过的对——组合泛化要看的正是这一栏")
    print("gain 正 = 比边际更准；unseen 上仍为正才叫组合泛化")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
