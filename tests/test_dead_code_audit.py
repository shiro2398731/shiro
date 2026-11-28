r"""死代码审计的测试（2026-11-28）。

Run: .\.venv\Scripts\python.exe tests\test_dead_code_audit.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.dead_code_audit import audit, collect_defs, collect_refs  # noqa: E402


def _make_tree(tmp: pathlib.Path) -> None:
    (tmp / "src").mkdir()
    (tmp / "tests").mkdir()
    (tmp / "tools").mkdir()
    (tmp / "src" / "a.py").write_text(
        "def used():\n    return 1\n\n"
        "def unused():\n    return 2\n\n"
        "class Used:\n    pass\n\n"
        "class Unused:\n    pass\n",
        encoding="utf-8",
    )
    (tmp / "tests" / "t.py").write_text(
        "from a import used, Used\nused()\nUsed()\n", encoding="utf-8")
    (tmp / "tools" / "x.py").write_text("", encoding="utf-8")


def test_audit_finds_unreferenced() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        _make_tree(tmp)
        defs = collect_defs(tmp)
        refs = collect_refs(tmp)
        assert refs["used"] >= 1 and refs["Used"] >= 1
        rows = audit(tmp)
        names = {r[0] for r in rows}
        assert "unused" in names, names
        assert "Used" not in names, names
        print(f"       unused candidates={sorted(names)}")


def test_defs_exclude_dunder() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = pathlib.Path(d)
        (tmp / "src").mkdir()
        (tmp / "src" / "b.py").write_text("def __init__():\n    pass\n", encoding="utf-8")
        assert all(not n.startswith("__") for n, _, _ in collect_defs(tmp))
        print("       dunder excluded")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
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
