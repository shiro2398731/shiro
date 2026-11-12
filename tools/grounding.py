"""
tools/grounding.py — 报告她的符号现在指着什么。

这是 `src/L5_cognition/grounding.py` 的消费者：在一份真实存档上跑 N 步，
把"符号 ↔ 指代"的统计量算出来，并给出零假设（把指代序列打乱）作对照。

用法
----
    python tools\\grounding.py                 # 3000 步，默认种子
    python tools\\grounding.py --steps 5000
    python tools\\grounding.py --json          # 只输一行 JSON，便于比对

注意
----
1. 只读存档，写回是关闭的：脚本不会调用任何 save_*。
2. 噪声幅度固定为 --amplitude（默认 0.02）：这是探针，不是她的运行环境；
   上游（P1 的惊讶驱动）不在这里，所以曲线与线上不同。
3. 两个表同时算：`stream.grounding` 由引擎内部在 step() 里累积，
   `probe` 由本脚本自己累积。两者应当逐位一致——不一致就说明接线有问题，
   所以这一步是自检，不是装饰。
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
from src.L5_cognition.grounding import GroundingTable  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def run(steps: int, amplitude: float = 0.02, seed: int = 0) -> dict:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive: run P2 first, or point state_dir at one")

    probe = GroundingTable()
    rows: list[tuple[int, tuple[int, ...]]] = []

    for _ in range(steps):
        st.landscape.set_noise_amplitude(amplitude)
        st.step()
        # Rebuild the same pair the engine saw, for the null hypothesis.
        speed = Dynamics.speed(st._st)
        tension = float(st.drive.summary().get("sum", 0.0))
        b_speed = probe.bucket("speed", speed)
        b_tension = probe.bucket("tension", tension)
        aid = st.detector.current_attractor
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
        "mi_engine_bits": round(mi_engine, 6),
        "mi_probe_bits": round(mi_probe, 6),
        "mi_shuffled_bits": round(mi_null, 6),
        "ratio_over_null": round(mi_probe / mi_null, 2) if mi_null > 0 else None,
        "coverage": round(probe.coverage(), 4),
        "top_symbols": top,
        "wiring_consistent": abs(mi_engine - mi_probe) < 1e-9,
    }
    return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="report what her symbols refer to")
    ap.add_argument("--steps", type=int, default=3000)
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
    print(f"MI(engine)={r['mi_engine_bits']:.4f} bits  "
          f"MI(probe)={r['mi_probe_bits']:.4f} bits  "
          f"MI(shuffled)={r['mi_shuffled_bits']:.4f} bits  "
          f"ratio={r['ratio_over_null']}x")
    print(f"wiring consistent (engine == probe): {r['wiring_consistent']}")
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
