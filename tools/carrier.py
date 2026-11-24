"""
tools/carrier.py — 把「她」装进一个可搬的文件，再从文件里装回来。

为什么需要它
------------
`src/common/carrier.py` 早就写好了格式（zip + schema + checksum），
但只有测试在用：没有任何一条命令让「可搬」这件事被真的做到过一次。
这个文件补上那条命令，并且**默认不覆盖活存档**——导入只会写一个新文件名，
要启用得由人手动改名。

用法
----
    python tools\\carrier.py export  [--out data/state/her.carrier]
    python tools\\carrier.py verify  <carrier> [--dim 128]
    python tools\\carrier.py import  <carrier> [--out-name l5.imported.pkl] [--force]
    python tools\\carrier.py describe <carrier>

判据
----
export 之后 verify 必须给出与导出时相同的 step_count / 区域数 / uid，
否则不算「同一个她」。verify 只在内存里推几步，不写任何状态文件。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_live(name: str = "l5.pkl"):
    from src.L5_cognition.stream import Stream
    from src.common import persist

    st = Stream()
    if not persist.load_l5(st, name=name):
        raise SystemExit(f"no archive {name}")
    return st


def cmd_export(args: argparse.Namespace) -> int:
    from src.common import carrier, persist

    st = _load_live(args.from_name)
    payload = persist.build_l5_payload(st)
    sp = st.landscape.state_space
    out = carrier.export(payload, args.out, extra_meta={
        "steps": int(st._step_count),
        "regions": len(sp.attractors()),
        "alive_uids": len(sp.alive_uids()),
        "born_seed": int(getattr(st, "_born_seed", 0) or 0),
        "source": "l5.pkl",
    })
    print(f"[carrier] exported {out} ({out.stat().st_size / 1024:.1f} KB)")
    print("[carrier] " + carrier.describe(out))
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    from src.common import carrier

    print(carrier.describe(args.carrier))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from src.L5_cognition.stream import Stream
    from src.common import carrier, persist

    payload = carrier.load(args.carrier, expect_dim=args.dim)
    st = Stream(dim=args.dim)
    persist.apply_l5_payload(st, payload)
    before = (int(st._step_count), len(st.landscape.state_space.attractors()),
              len(st.landscape.state_space.alive_uids()))
    # 走几步：证明装回来的不是一个只能打印的死结构。
    for _ in range(5):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    after = (int(st._step_count), len(st.landscape.state_space.attractors()),
             len(st.landscape.state_space.alive_uids()))
    print(f"[carrier] loaded: steps={before[0]} regions={before[1]} "
          f"uids={before[2]}")
    print(f"[carrier] after 5 steps: steps={after[0]} regions={after[1]} "
          f"uids={after[2]}")
    if after[0] != before[0] + 5:
        print("[carrier] FAIL: step count did not advance by 5")
        return 1
    print("[carrier] OK: the file is a working body, not a snapshot dump")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from src.L5_cognition.stream import Stream
    from src.common import carrier, persist

    name = args.out_name
    if name == "l5.pkl" and not args.force:
        print("[carrier] refusing to overwrite the live checkpoint "
              "(pass --force if you really mean it)")
        return 2
    payload = carrier.load(args.carrier, expect_dim=args.dim)
    st = Stream(dim=args.dim)
    persist.apply_l5_payload(st, payload)
    out = persist.save_l5(st, name=name)
    print(f"[carrier] imported -> {out} "
          f"({out.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"[carrier] steps={st._step_count} "
          f"regions={len(st.landscape.state_space.attractors())}")
    print("[carrier] not activated: rename it over l5.pkl yourself, "
          "only when no process is writing")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="move her as a file, verify, import")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export")
    e.add_argument("--out", default=os.path.join("data", "state", "her.carrier"))
    e.add_argument("--from", dest="from_name", default="l5.pkl")
    e.set_defaults(fn=cmd_export)

    d = sub.add_parser("describe")
    d.add_argument("carrier")
    d.set_defaults(fn=cmd_describe)

    v = sub.add_parser("verify")
    v.add_argument("carrier")
    v.add_argument("--dim", type=int, default=128)
    v.set_defaults(fn=cmd_verify)

    i = sub.add_parser("import")
    i.add_argument("carrier")
    i.add_argument("--out-name", default="l5.imported.pkl")
    i.add_argument("--dim", type=int, default=128)
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_import)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
