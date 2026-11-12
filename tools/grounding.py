"""
tools/grounding.py — 报告她的符号现在指着什么。

消费者：在真实存档上跑 N 步，把「符号 → 靶子」的统计量算出来。

判据只有一种：**留出法**（前一半拟合 p(靶子|符号)，后一半算对数损失）。
plug-in 互信息也打印，但标了 [biased]：它在样本稀疏时严重上偏，
实测同一条数据 0.6096 bits 对打乱后 0.0324（18.8x），留出法却是 −0.0517（比边际更差）。

两个符号定义：
    A = 当前区域
    B = (上一个区域, 当前区域)      —— 复合符号，用来回答「组合是否带来语义带宽」

两个靶子：
    state  = (speed 桶, tension 桶)      她此刻的内部状态量
    next   = 下一个区域                  她下一步去哪

实测（她的真实存档，4000 步）：

                 A 当前区域        B (前,现)
    state         -0.0517 (-1.2%)   -0.0004 (-0.0%)   无指代（两种符号都没有）
    next          +2.6099 (78.3%)   +2.5118 (75.4%)   有信息，复合符号反而更差

结论：符号编码的是**转移结构**，不是瞬时状态量；把符号换成复合的，
在转移靶上没有增益（更细的字母表换来更稀的格子）。详见输出。

用法
----
    python tools\\grounding.py                 # 4000 步，默认种子
    python tools\\grounding.py --steps 6000
    python tools\\grounding.py --json

注意
----
1. 只读存档，写回是关闭的：脚本不会调用任何 save_*。
2. 噪声幅度固定为 --amplitude（默认 0.02）：这是探针，不是她的运行环境。
3. `stream.grounding`（引擎内部累积）与 `probe`（本脚本累积）必须逐位一致，
   不一致说明接线有问题 —— 这一项是自检。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.dynamics import Dynamics  # noqa: E402
from src.L5_cognition.grounding import GroundingTable, heldout_gain  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def run(steps: int, amplitude: float = 0.02, seed: int = 0) -> dict:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive: run P2 first, or point state_dir at one")

    probe = GroundingTable()
    recs: list = []                    # (current region, previous region, context or None)

    for _ in range(steps):
        st.landscape.set_noise_amplitude(amplitude)
        st.step()
        speed = Dynamics.speed(st._st)
        tension = float(st.drive.summary().get("sum", 0.0))
        b_speed = probe.bucket("speed", speed)
        b_tension = probe.bucket("tension", tension)
        aid = st.detector.current_attractor
        if aid is None:
            continue
        ctx = (b_speed, b_tension) if (b_speed is not None and b_tension is not None) else None
        prev = recs[-1][0] if recs else None
        recs.append((int(aid), prev, ctx))
        if ctx is not None:
            probe.observe(int(aid), ctx)

    mi_engine = st.grounding.mutual_information()
    mi_probe = probe.mutual_information()

    ctx_rows = [(c, ctx) for c, _p, ctx in recs if ctx is not None]
    ctx_only = [ctx for _c, ctx in ctx_rows]
    random.Random(seed).shuffle(ctx_only)
    null = GroundingTable()
    for (c, _), ctx in zip(ctx_rows, ctx_only):
        null.observe(c, ctx)
    mi_null = null.mutual_information()

    # ---- the four held-out evaluations: {A, B} symbols x {state, next} targets ----
    state_a = [(c, ctx) for c, _p, ctx in recs if ctx is not None]
    state_b = [((p, c) if p is not None else None, ctx)
               for c, p, ctx in recs if ctx is not None]
    trans_a = [(recs[i][0], recs[i + 1][0]) for i in range(len(recs) - 1)]
    trans_b = [((recs[i][1], recs[i][0]) if recs[i][1] is not None else None,
                recs[i + 1][0]) for i in range(len(recs) - 1)]

    heldout = {
        "state_A": heldout_gain(state_a),
        "state_B": heldout_gain(state_b),
        "next_A": heldout_gain(trans_a),
        "next_B": heldout_gain(trans_b),
    }

    symbols = sorted(probe.symbol_totals, key=lambda s: -probe.symbol_totals[s])
    top = [{"symbol": int(s), "count": int(probe.symbol_totals[s]),
            "label": probe.label(s), "specificity": round(probe.specificity(s), 4)}
           for s in symbols[:8]]

    return {
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
    print(f"plug-in MI={r['mi_probe_bits']:.4f} bits  shuffled={r['mi_shuffled_bits']:.4f}"
          f"  ratio={r['ratio_over_null']}x   [biased; judge by held-out below]")
    print(f"wiring consistent (engine == probe): {r['wiring_consistent']}")

    ho = r["heldout"]
    print("held-out gain, bits/observation (positive = the symbol helps):")
    print(f"  {'':<10}{'A current':>14}{'B (prev,cur)':>16}")
    for key, name in (("state", "state"), ("next", "next region")):
        a = ho.get(f"{key}_A") or {}
        b = ho.get(f"{key}_B") or {}
        cells = []
        for d in (a, b):
            if not d.get("n_test"):
                cells.append(f"{'n/a':>14}")
            else:
                share = 100.0 * (d.get("gain_share") or 0.0)
                cells.append(f"{d['gain_bits']:+.4f} ({share:+.1f}%)".rjust(14))
        print(f"  {name:<10}{cells[0]}{cells[1]:>16}")
    print(f"  (n_test per cell: state {ho['state_A'].get('n_test')}, "
          f"next {ho['next_A'].get('n_test')})")
    print("NOTE: zero/negative gain means the table memorised the sample;")
    print("      that symbol does not refer to that target.")

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
