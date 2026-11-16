r"""tools/bench.py — 单步成本的常驻基准。

为什么要有这个
--------------
2026-11-16：文档里记的单步成本是 6.20 ms，实测已经是 18–25 ms（随存档大小浮动），
差了三四倍，但没有任何一项测试或工具会为此报警。改了一个「显然安全」的优化之后，
性能没变、语义反而变了（见 docs/2026-11-16_性能_sigma缓存.md）。所以先把
「跑一步要多久」变成一个随时能跑出一个数的东西。

它不做判断，只出数。判断在别处。

用法：
    python tools\bench.py                     # 默认 120 步
    python tools\bench.py --steps 300
    python tools\bench.py --json

注意：载入的是 data/state 下的存档，只读、不回写；它跑在自己的进程里，不会碰
正在运行的 P2。数字会随存档大小浮动（区域越多越慢），比较时要拿同一份存档。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def run(steps: int = 120, amplitude: float = 0.02, warmup: int = 10) -> dict:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive: run P2 first, or point state_dir at one")

    space = st.landscape.state_space
    start = {
        "regions": len(space.attractors()),
        "edges": len(st.graph),
        "step_count": int(st._step_count),
        "x_sha": hashlib.sha256(st._st.x.tobytes()).hexdigest()[:16],
    }

    for _ in range(warmup):
        st.landscape.set_noise_amplitude(amplitude)
        st.step()

    t0 = time.perf_counter()
    for _ in range(steps):
        st.landscape.set_noise_amplitude(amplitude)
        st.step()
    elapsed = time.perf_counter() - t0

    end = {
        "regions": len(space.attractors()),
        "edges": len(st.graph),
        "step_count": int(st._step_count),
        "x_sha": hashlib.sha256(st._st.x.tobytes()).hexdigest()[:16],
    }
    return {
        "steps": int(steps),
        "warmup": int(warmup),
        "amplitude": float(amplitude),
        "ms_per_step": round(elapsed / steps * 1000.0, 3),
        "steps_per_sec": round(steps / elapsed, 2) if elapsed > 0 else None,
        "start": start,
        "end": end,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="single-step cost, read-only")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--amplitude", type=float, default=0.02)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    r = run(args.steps, args.amplitude, args.warmup)
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    s, e = r["start"], r["end"]
    print(f"steps={r['steps']}  warmup={r['warmup']}  amplitude={r['amplitude']}")
    print(f"  {r['ms_per_step']:.2f} ms/step  ({r['steps_per_sec']} steps/s)")
    print(f"  start: regions={s['regions']} edges={s['edges']} "
          f"step_count={s['step_count']} x_sha={s['x_sha']}")
    print(f"  end  : regions={e['regions']} edges={e['edges']} "
          f"step_count={e['step_count']} x_sha={e['x_sha']}")
    print("  x_sha 是终点状态的指纹：同一份存档、同一段代码跑出来的必须相同。")
    print("  改了任何影响动力学的代码之后，它变了就说明语义变了，不管 ms/step 好看多少。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
