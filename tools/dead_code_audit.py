r"""死代码审计（2026-11-28）。

只读：扫描 src 下的顶层 def/class，统计在 src/tests/tools 里被引用的次数，
列出「零引用候选」。不自动删——候选里可能有动态引用/入口点，要人看。

用法：.\.venv\Scripts\python.exe tools/dead_code_audit.py
退出码：0（只报告）。
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def collect_defs(root: Path) -> list[tuple[str, str, int]]:
    out = []
    for p in sorted((root / "src").rglob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("__"):
                    continue
                out.append((node.name, str(p.relative_to(root)), node.lineno))
    return out


def collect_refs(root: Path) -> Counter:
    refs: Counter = Counter()
    for base in ("src", "tests", "tools"):
        for p in (root / base).rglob("*.py"):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    refs[node.id] += 1
                elif isinstance(node, ast.Attribute):
                    refs[node.attr] += 1
    return refs


def audit(root: Path, min_refs: int = 1) -> list[tuple[str, str, int, int]]:
    defs = collect_defs(root)
    refs = collect_refs(root)
    out = []
    for name, path, line in defs:
        # def 行本身会算 1 次（ast.Name 里不含定义名，所以不会）
        n = refs.get(name, 0)
        if n < min_refs:
            out.append((name, path, line, n))
    return out


def main() -> int:
    rows = audit(ROOT)
    print("== 零引用候选（{} 个）==".format(len(rows)))
    for name, path, line, n in rows:
        print(f"  {path}:{line}: {name}  refs={n}")
    print("\n注意：入口点、动态调用、协议方法会被误报；删之前逐个看。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
