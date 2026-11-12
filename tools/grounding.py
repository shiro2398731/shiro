"""
tools/grounding.py — 报告她的符号现在指着什么。

这是 `src/L5_cognition/grounding.py` 的消费者：在一份真实存档上跑 N 步，
把"符号 ↔ 指代"的统计量算出来，并给出三个对照：

  1. 零假设：把指代序列打乱后重算的 plug-in MI（有限样本偏置的下界）；
  2. **留出法（状态靶）**：前一半拟合 p(speed/tension 桶 | 符号)，后一半评估对数损失；
  3. **留出法（转移靶）**：前一半拟合 p(下一个区域 | 当前区域)，后一半评估对数损失。

第 2 项才是"符号是否贴着某个内在标量"的判据；第 3 项是"符号是否带着关于她下一步的信息"的判据。

实测（她的真实存档，4000 步，2000 条留出样本）：

    状态靶  baseline 4.1580 → model 4.2096 bits   gain −0.0517   （无增益，表只是记住了样本）
    转移靶  baseline 3.3325 → model 0.7226 bits   gain +2.6099   （H(next)=3.256，消掉 78.3%）

所以现在能声称的是：**符号没有编码瞬时状态量，但编码了转移结构**。
"意义来自它与别的符号的关系"在这里第一次有了正的数字支持；
而"某个区域贴着某个内在标量"这件事，用留出法测不出来（plug-in MI 0.61–0.74 bits 是偏置）。

用法
----
    python tools\\grounding.py                 # 4000 步，默认种子
    python tools\\grounding.py --steps 6000
    python tools\\grounding.py --json          # 只输一行 JSON，便于比对

注意
----
1. 只读存档，写回是关闭的：脚本不会调用任何 save_*。
2. 噪声幅度固定为 --amplitude（默认 0.02）：这是探针，不是她的运行环境；
   上游（P1 的惊讶驱动）不在这里，所以曲线与线上不同。
3. 两个表同时算：`stream.grounding` 由引擎内部在 step() 里累积，
   `probe` 由本脚本自己累积。两者应当逐位一致——不一致就说明接线有问题。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

# Laplace smoothing for the held-out tests. C1: it keeps unseen cells from
# producing an infinite loss; it does not change what is being measured.
ALPHA = 0.5

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.dynamics import Dynamics  # noqa: E402
from src.L5_cognition.grounding import GroundingTable  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def _heldout(pairs: list, keys: list, alpha: float = ALPHA) -> dict:
    """Held-out log-loss of p(y | symbol) vs p(y), for one symbol definition.

    `pairs` is [(symbol, y), ...] in time order; halves are used as train/test.
    """
    split = len(pairs) // 2
    train, test = pairs[:split], pairs[split:]
    if not train or not test:
        return {"n_test": 0}
    ys = sorted({y for _, y in train})
    base = collections.Counter(y for _, y in train)
    n_train = sum(base.values())
    p_base = {y: (base[y] + alpha) / (n_train + alpha * len(ys)) for y in ys}

    row: dict = collections.defaultdict(collections.Counter)
    tot: collections.Counter = collections.Counter()
    for s, y in train:
        if s is None:
            continue
        row[s][y] += 1
        tot[s] += 1

    ll_base = ll_model = 0.0
    n = 0
    for s, y in test:
        p = p_base[y]
        if s is not None and tot.get(s):
            p = (row[s][y] + alpha) / (tot[s] + alpha * len(ys))
        ll_base += -math.log2(p_base[y])
        ll_model += -math.log2(p)
        n += 1
    h_y = -sum((v / n_train) * math.log2(v / n_train) for v in base.values())
    return {
        "n_test": n,
        "distinct_targets": len(ys),
        "h_target_bits": round(h_y, 4),
        "baseline_bits": round(ll_base / n, 4),
        "model_bits": round(ll_model / n, 4),
        "gain_bits": round((ll_base - ll_model) / n, 4),
        "gain_share": round((ll_base - ll_model) / ll_base, 4) if ll_base else None,
    }


def run(steps: int, amplitude: float = 0.02, seed: int = 0) -> dict:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive: run P2 first, or point state_dir at one")

    probe = GroundingTable()
    rows: list[tuple[int, tuple[int, ...]]] = []
    seq: list[int] = []

    for _ in range(steps):
        st.landscape.set_noise_amplitude(amplitude)
        st.step()
        speed = Dynamics.speed(st._st)
        tension = float(st.drive.summary().get("sum", 0.0))
        b_speed = probe.bucket("speed", speed)
        b_tension = probe.bucket("tension", tension)
        aid = st.detector.current_attractor
        if aid is not None:
            seq.append(int(aid))
        if aid is not None and b_speed is not None and b_tension is not None:
            ctx = (b_speed, b_tension)
            probe.observe(int(aid), ctx)
            rows.append((int(aid), ctx))

    mi_engine = st.grounding.mutual_information()
    mi_probe = probe.mutual_information()

    ctx_only = [c for _, c in rows]
    random.Random(seed).shuffle(ctx_only)
    null = GroundingTable()
    for (s, _), c in zip(rows, ctx_only):
        null.observe(s, c)
    mi_null = null.mutual_information()

    # ---- held-out evaluations ----
    state_pairs = [(a, c) for a, c in rows]
    comp_pairs = []
    prev = None
    for a, c in rows:
        comp_pairs.append(((prev, a) if prev is not None else None, c))
        prev = a
    next_pairs = list(zip(seq, seq[1:]))

    heldout = {
        "state_target": _heldout(state_pairs, ["A"]),
        "state_target_composite": _heldout(comp_pairs, ["B"]),
        "transition_target": _heldout(next_pairs, ["T"]),
    }

    symbols = sorted(probe.symbol_totals, key=lambda s: -probe.symbol_totals[s])
    top = [{"symbol": int(s), "count": int(probe.symbol_totals[s]),
            "label": probe.label(s), "specificity": round(probe.specificity(s), 4)}
           for s in symbols[:8]]

    report = {
        "steps": int(steps),
        "amplitude": float(amplitude),
        "observations": int(probe.total),
        "symbols": len(probe.symbol_totals),
        "contexts": len(probe.context_totals),
        "h_symbol_bits": round(probe.entropy_symbol(), 6),
        "h_context_bits": round(probe.entropy_context(), 6),
        "mi_engine_bits": round(mi_engine, 6),
        "mi_probe_bits": round(mi_probe, 6),
        "mi_shuffled_bits": round(mi_null, 6),
        "ratio_over_null": round(mi_probe / mi_null, 2) if mi_null > 0 else None,
        "coverage": round(probe.coverage(), 4),
        "top_symbols": top,
        "wiring_consistent": abs(mi_engine - mi_probe) < 1e-9,
        "heldout": heldout,
    }
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="report what her symbols refer to")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--amplitude", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    r = run(args.steps, args.amplitude, args.seed)
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    print(f"steps={r['steps']}  amplitude={r['amplitude']}  "
          f"observations={r['observations']}")
    print(f"symbols={r['symbols']}  contexts={r['contexts']}  "
          f"coverage={r['coverage']:.2%}")
    print(f"H(symbol)={r['h_symbol_bits']:.4f} bits  "
          f"H(context)={r['h_context_bits']:.4f} bits")
    print(f"plug-in MI(engine)={r['mi_engine_bits']:.4f}  "
          f"MI(probe)={r['mi_probe_bits']:.4f}  "
          f"MI(shuffled)={r['mi_shuffled_bits']:.4f}  "
          f"ratio={r['ratio_over_null']}x   [biased; see below]")
    print(f"wiring consistent (engine == probe): {r['wiring_consistent']}")

    ho = r["heldout"]
    print("held-out log-loss (lower is better; gain > 0 means the symbol helps):")
    labels = (
        ("state_target", "A  region -> speed/tension"),
        ("state_target_composite", "B  (prev,cur) -> speed/tension"),
        ("transition_target", "T  region -> next region"),
    )
    for key, name in labels:
        d = ho.get(key) or {}
        if not d.get("n_test"):
            continue
        print(f"  {name:<30} symbols/targets={d['distinct_targets']:>3}  "
              f"H(y)={d['h_target_bits']:.4f}  baseline={d['baseline_bits']:.4f}  "
              f"model={d['model_bits']:.4f}  gain={d['gain_bits']:+.4f} bits "
              f"({100.0 * (d['gain_share'] or 0):+.1f}%)")
    print("NOTE: positive plug-in MI with zero held-out gain means the table")
    print("      memorised the sample; that symbol does not refer to that target.")

    print("top symbols by count:")
    for t in r["top_symbols"]:
        print(f"  region {t['symbol']:>4}  n={t['count']:>5}  "
              f"referent={t['label']:<8}  specificity={t['specificity']}")
    if not r["wiring_consistent"]:
        print("MISMATCH: the engine's table and the probe disagree; fix the wiring",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
