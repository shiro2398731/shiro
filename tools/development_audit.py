r"""发育审计（只读）：她从旧存档到今天，结构有没有不可逆地长。

比较两份 l5 存档（旧 → 新）：
- 区域 uid 的新增/保留/消失；
- 图的边数与节点保留；
- 身份核心的 Jaccard；
- 动作分布。

Run: .\.venv\Scripts\python.exe tools\development_audit.py <old.pkl> [new.pkl]
默认 new = l5.pkl。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from src.L8_identity.identity import Identity  # noqa: E402


def load(name: str):
    st = Stream()
    assert persist.load_l5(st, name=name), name
    return st


def graph_nodes(st) -> set[int]:
    nodes = set()
    for e in st.graph.edges():
        nodes.add(int(e.a))
        nodes.add(int(e.b))
    return nodes


def core_nodes(st) -> set[int]:
    ident = Identity()
    core = ident.observe(st.graph)
    return set(core.nodes()) if not core.is_empty() else set()


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: development_audit.py <old.pkl> [new.pkl]")
        return 2
    old_name, new_name = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "l5.pkl")
    old, new = load(old_name), load(new_name)
    old_uids = {int(a.uid) for a in old.landscape.state_space.attractors()}
    new_uids = {int(a.uid) for a in new.landscape.state_space.attractors()}
    old_g, new_g = graph_nodes(old), graph_nodes(new)
    old_c, new_c = core_nodes(old), core_nodes(new)
    res = {
        "old": old_name, "new": new_name,
        "regions_old": len(old_uids), "regions_new": len(new_uids),
        "uid_kept": len(old_uids & new_uids),
        "uid_new": len(new_uids - old_uids),
        "uid_gone": len(old_uids - new_uids),
        "edges_old": len(old.graph), "edges_new": len(new.graph),
        "graph_nodes_old": len(old_g), "graph_nodes_new": len(new_g),
        "graph_node_jaccard": round(jaccard(old_g, new_g), 4),
        "core_old": len(old_c), "core_new": len(new_c),
        "core_jaccard": round(jaccard(old_c, new_c), 4),
        "actions_old": sum(old.action_counts.values()),
        "actions_new": sum(new.action_counts.values()),
    }
    print("== 发育审计（只读）==")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
