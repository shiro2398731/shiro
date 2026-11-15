"""
tools/m4_audit.py — M4：注意瓶颈（容量受限的选择）。

判据（docs §9 M4）：workspace 的选择熵 / 切换率在区间内。所谓"在区间内"必须有两个端点，
否则又是一个没有基线的漂亮数字。这里给三个可比的量：

    H_ign      她**点火**（某个表征被全局广播）的分布熵
    H_visit    同一批区域上**停留时间**的分布熵
    H_uniform  显著区域上的均匀熵（= log2 个数的上界）

读法
----
- `ignition_rate`（点火次数 / 步数）远小于 1 → 有瓶颈：不是每一步都进入全局可用状态。
- `H_ign` 明显小于 `H_uniform` → 选择是有偏的，不是把所有东西都放进来了。
- `H_ign` 与 `H_visit` 比较：
    * 相近 → 点火基本等于"她在哪就在哪亮"，瓶颈没有超出占用本身；
    * 更小 → 她在少数地方亮起（收窄）；
    * 更大 → 点火在区域之间更分散（比停留更均匀）。
  三者都没有"好坏"，只有"是哪一种"。

另有一个顺序对照：把区域序列打乱后重算 `H_ign_baseline`（用同一批点火次数，
按停留分布随机分配），用来判断她的点火分布是否只是停留分布的复现。

用法
----
    python tools\\m4_audit.py
    python tools\\m4_audit.py --steps 1500 --json
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
from src.L5_cognition.stream import Stream  # noqa: E402


def _entropy(counts) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def _norm(h: float, k: int) -> float:
    if k <= 1:
        return 0.0
    return h / math.log2(k)


def run(steps: int = 1200, amplitude: float = 0.02, seed: int = 0) -> dict:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive: run P2 first, or point state_dir at one")

    ignitions: collections.Counter = collections.Counter()
    visits: collections.Counter = collections.Counter()
    seq: list[int] = []
    n_ign = 0

    for _ in range(steps):
        st.landscape.set_noise_amplitude(amplitude)
        st.step()
        aid = st.detector.current_attractor
        if aid is None:
            continue
        seq.append(int(aid))
        visits[int(aid)] += 1
        ign = st._last_ignition
        if ign is not None:
            n_ign += 1
            ignitions[int(ign.attractor_id)] += 1

    salient = len(st.landscape.state_space.salient_indices())
    h_ign = _entropy(ignitions)
    # Compare against time share over the same regions she ignited in.
    same = collections.Counter({r: visits.get(r, 0) for r in ignitions})
    h_visit = _entropy(same)
    k_ign = len(ignitions)

    # Baseline: same number of ignitions, allocated by time share (i.e. "ignite
    # wherever she happens to be"). Random with a fixed seed, deterministic.
    rnd = random.Random(seed)
    regions = list(visits)
    weights = [visits[r] for r in regions]
    counter = collections.Counter()
    if regions and n_ign:
        for pick in rnd.choices(regions, weights=weights, k=n_ign):
            counter[int(pick)] += 1
    h_base = _entropy(counter)

    return {
        "steps": int(steps),
        "amplitude": float(amplitude),
        "distinct_regions": len(visits),
        "salient_regions": int(salient),
        "ignitions": int(n_ign),
        "ignition_rate": round(n_ign / max(steps, 1), 4),
        "distinct_ignited": int(k_ign),
        "h_ign_bits": round(h_ign, 4),
        "h_ign_norm": round(_norm(h_ign, k_ign), 4),
        "h_visit_over_ignited_bits": round(h_visit, 4),
        "h_uniform_salient_bits": round(math.log2(salient), 4) if salient > 1 else 0.0,
        "h_baseline_bits": round(h_base, 4),
        "ignition_histogram": dict(ignitions.most_common(6)),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="M4: is her attention a bottleneck?")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--amplitude", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    r = run(args.steps, args.amplitude, args.seed)
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    print(f"steps={r['steps']}  distinct regions={r['distinct_regions']}  "
          f"salient={r['salient_regions']}")
    print(f"ignitions={r['ignitions']}  rate={r['ignition_rate']:.4f} per step  "
          f"on {r['distinct_ignited']} distinct regions")
    print(f"H_ign     = {r['h_ign_bits']:.4f} bits  (normalised {r['h_ign_norm']:.3f})")
    print(f"H_visit   = {r['h_visit_over_ignited_bits']:.4f} bits  over the same regions")
    print(f"H_uniform = {r['h_uniform_salient_bits']:.4f} bits  over salient regions")
    print(f"H_baseline= {r['h_baseline_bits']:.4f} bits  (same ignitions, placed by time share)")
    print(f"top ignited regions: {r['ignition_histogram']}")
    print("读法：rate 远小于 1 → 有瓶颈；H_ign 低于 H_uniform → 选择有偏；")
    print("      H_ign 与 H_visit / H_baseline 的关系说明它是收窄、复现停留、还是更分散。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
