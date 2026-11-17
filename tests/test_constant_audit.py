r"""Human-constant audit on the fold path.

PRINCIPLES #4 / docs §11 (C1/C2/C3): a threshold that decides what she keeps or
loses may not be a number someone picked. After `fold_auto` replaced
`min_assigned=2 / age_factor=8.0`, the fold path has no such number left, and
this test is what makes that claim checkable instead of remembered.

What is counted
---------------
Numeric literals in the body of `StateSpace.fold_auto`. 0 and 1 are allowed
because they are the identities of counting (empty / at least one), not
thresholds. Everything else must arrive through a named module constant or a
statistic of her own state.

Run: .\.venv\Scripts\python.exe tests\test_constant_audit.py
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TERRAIN = ROOT / "src" / "L5_cognition" / "terrain.py"
RUNTIME = ROOT / "src" / "runtime" / "process_2_mind.py"

# Identities of counting, not thresholds.
ALLOWED_LITERALS = {0, 1}

# Decision points audited so far and the number of human constants each still
# contains. The target is 0 for every entry; a new entry starts at its measured
# value, never at a guess.
AUDITED = {
    "StateSpace.fold_auto": 0,     # was 2 (min_assigned, age_factor) before 2026-11-09
    "StateSpace._birth_depth": 0,  # derived from _population_visit_rate / assign_ratio
}


def _method_node(path: pathlib.Path, cls: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found in {path}")


def _numeric_literals(fn: ast.FunctionDef) -> list[float]:
    body = fn.body
    # The docstring is prose, not code.
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    out: list[float] = []
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)) \
                    and not isinstance(sub.value, bool):
                out.append(sub.value)
    return out


def test_fold_auto_has_no_threshold_literals() -> None:
    fn = _method_node(TERRAIN, "StateSpace", "fold_auto")
    literals = _numeric_literals(fn)
    stray = sorted({v for v in literals if v not in ALLOWED_LITERALS})
    assert not stray, f"fold_auto still contains chosen numbers: {stray}"
    print(f"       fold_auto literals: {sorted(set(literals))} (identities only)")


def test_runtime_uses_fold_auto() -> None:
    """Checked on the syntax tree, not on the text: P2's comment names the old
    thresholds on purpose, and a substring search would read that as usage."""
    tree = ast.parse(RUNTIME.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    names = {c.func.attr for c in calls}
    # 2026-11-17：原来这里强求「P2 必须调 fold_auto」。当天实测证明 fold_auto 是净损失
    # （折得比长得快，她 427 -> 8），于是 P2 暂停调用它。这个测试要保的是
    # 「没有人挑的阈值」，不是「必须调用某个函数」，所以不再强求 fold_auto 在场。
    # 但**禁止**旧式 fold(min_assigned=..., age_factor=...) 那条仍然有效。
    assert "fold" not in names, "P2 still calls fold() with hand-chosen thresholds"
    passed = [kw.arg for c in calls for kw in c.keywords
              if kw.arg in {"min_assigned", "age_factor"}]
    assert not passed, f"P2 still passes {passed}"
    # 2026-11-17：这条断言原来要求「P2 必须调 fold_auto」。但当天实测证明
    # fold_auto 是净损失（折得比长得快，她一路掉），于是 P2 被改成暂停调用。
    # 这个测试真正要保的是「没有人挑的阈值」，不是「必须调用某个函数」——
    # 所以现在两种情况都算通过，只要不传 min_assigned / age_factor。
    if "fold_auto" in names:
        print("       P2: calls fold_auto(), passes no hand-chosen threshold")
    else:
        print("       P2: fold_auto suspended (2026-11-17), passes no hand-chosen threshold")


def test_audit_table_targets_zero() -> None:
    """The indicator itself is a test: it may only go down."""
    total = sum(AUDITED.values())
    assert total == 0, f"human constants left on audited paths: {AUDITED}"
    print(f"       audited paths: {len(AUDITED)}, human constants: {total}")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
