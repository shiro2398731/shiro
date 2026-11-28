r"""L8→L5 审计：身份核心随时间的稳定性（2026-11-28）。

只读副本。每步用一个新的 Identity 观察她的痕迹图（不碰她自己的 identity）：
- 核心节点数、边的数量；
- 相邻两步核心的 Jaccard（身份稳不稳）；
- 空核心比例；
- fingerprint 维度。

背景：`landscape.set_self_bias()` 从未被调用（L8→L5 死通道），
这个审计量的是「身份本身有没有稳定的东西可接」。

Run: .\.venv\Scripts\python.exe tools\l8_identity_audit.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from src.L8_identity.identity import Identity  # noqa: E402

STEPS = 400
CP = Path("data/state/_audit_l8.pkl")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main() -> int:
    shutil.copyfile(Path("data/state/l5.pkl"), CP)
    st = Stream()
    assert persist.load_l5(st, name=CP.name)
    ident = Identity()
    rows = []
    prev = set()
    for _ in range(STEPS):
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        core = ident.observe(st.graph)
        nodes = set(core.nodes()) if not core.is_empty() else set()
        rows.append({
            "nodes": len(nodes),
            "edges": core.describe(),
            "empty": bool(core.is_empty()),
            "jaccard": jaccard(prev, nodes),
        })
        prev = nodes
    try:
        CP.unlink()
    except FileNotFoundError:
        pass
    fp = ident.fingerprint(st.landscape.state_space, st.graph)
    sizes = [r["nodes"] for r in rows]
    jac = [r["jaccard"] for r in rows]
    out = {
        "steps": len(rows),
        "core_nodes_mean": round(sum(sizes) / max(len(sizes), 1), 2),
        "core_nodes_max": max(sizes) if sizes else 0,
        "empty_share": round(sum(1 for r in rows if r["empty"]) / max(len(rows), 1), 3),
        "jaccard_mean": round(sum(jac) / max(len(jac), 1), 3),
        "jaccard_min": round(min(jac), 3) if jac else None,
        "fingerprint_dims": len(fp),
        "rounds": ident.rounds,
    }
    dest = Path("data/exp/l8_identity_audit.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"summary": out, "rows": rows}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print("== L8 身份审计（副本 %d 步）==" % STEPS)
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"  明细: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
