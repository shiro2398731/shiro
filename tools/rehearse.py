"""
tools/rehearse.py — 预演：把当下向前推 k 步，然后看看错在哪。

和"重放"不同，这个工具不重建过去，而是**从已知算出未知**：她现在在这里，
如果用她自己的动力学往前推 k 步，会到哪？然后真的走 k 步，比较。

对照（缺了这个就是自欺）：
    stay       "她哪也不去"（预测终点 = 当前区域）
    marginal   她预热期最常待的区域
    预演        副本上的 k 步推演

同时给三件东西：
1. 每个 k 的终点命中率与基线；
2. 每一步的命中率（预演序列 vs 实际序列）——看预演是"慢慢漂走"还是"一开始就错"；
3. 反事实：从同一刻出发，强制 do(hold) 与 do(leave) 各推 k 步，比较落点的差异。
   这是"记忆的另一种处理方式"：不是回想，而是问"如果我当时那样做"。

用法
----
    python tools\\rehearse.py
    python tools\\rehearse.py --steps 600 --horizons 1,4,16,64
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.action import HOLD, LEAVE  # noqa: E402
from src.L5_cognition.sandbox import advance, project  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def _fresh(amplitude: float) -> Stream:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive: run P2 first, or point state_dir at one")
    st.landscape.set_noise_amplitude(amplitude)
    return st


def run(warm: int = 400, horizons=(1, 4, 16, 64), amplitude: float = 0.02,
        trials: int = 6) -> dict:
    out = {"warm": warm, "amplitude": amplitude, "horizons": list(horizons),
           "trials": int(trials), "rows": []}

    for k in horizons:
        hit_pred = hit_stay = hit_marg = 0
        per_step = [0] * k
        n = 0
        for _ in range(trials):
            st = _fresh(amplitude)
            seen = collections.Counter()
            for _ in range(warm):
                st.landscape.set_noise_amplitude(amplitude)
                st.step()
                aid = st.detector.current_attractor
                if aid is not None:
                    seen[int(aid)] += 1
            here = st.detector.current_attractor
            here = -1 if here is None else int(here)
            top = seen.most_common(1)[0][0] if seen else -1

            proj = project(st, k, amplitude)["regions"]
            real = advance(st, k, amplitude)

            hit_pred += int(proj[-1] == real[-1])
            hit_stay += int(here == real[-1])
            hit_marg += int(top == real[-1])
            for i in range(k):
                per_step[i] += int(proj[i] == real[i])
            n += 1

        out["rows"].append({
            "k": int(k),
            "trials": n,
            "pred_hit": round(hit_pred / n, 4),
            "stay_hit": round(hit_stay / n, 4),
            "marginal_hit": round(hit_marg / n, 4),
            "per_step_hit": [round(v / n, 4) for v in per_step],
        })
    return out


def counterfactual(warm: int = 400, k: int = 16, amplitude: float = 0.02,
                   trials: int = 6) -> dict:
    same = 0
    n = 0
    for _ in range(trials):
        st = _fresh(amplitude)
        for _ in range(warm):
            st.landscape.set_noise_amplitude(amplitude)
            st.step()
        a = project(st, k, amplitude, force_action=HOLD)["final"]
        b = project(st, k, amplitude, force_action=LEAVE)["final"]
        same += int(a == b)
        n += 1
    return {"horizon": int(k), "trials": n, "same_landing": round(same / n, 4)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="rehearse: project forward in a sandbox, then check")
    ap.add_argument("--warm", type=int, default=400)
    ap.add_argument("--horizons", type=str, default="1,4,16,64")
    ap.add_argument("--amplitude", type=float, default=0.02)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    r = run(args.warm, horizons, args.amplitude, args.trials)
    cf = counterfactual(args.warm, max(horizons), args.amplitude, args.trials)
    r["counterfactual"] = cf

    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    print(f"warm={r['warm']}  amplitude={r['amplitude']}  trials={r['trials']}")
    print(f"  {'k':>4}{'预演命中':>10}{'原地不动':>10}{'最常待':>10}")
    for row in r["rows"]:
        print(f"  {row['k']:>4}{row['pred_hit']:>10.3f}{row['stay_hit']:>10.3f}"
              f"{row['marginal_hit']:>10.3f}")
    print(f"反事实（do(hold) vs do(leave)，k={cf['horizon']}）："
          f"落点相同的比例 {cf['same_landing']:.3f}")

    print("逐步命中（k 最大的那一行）：")
    row = r["rows"][-1]
    step_txt = " ".join(f"{v:.2f}" for v in row["per_step_hit"][:16])
    print(f"  {step_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
