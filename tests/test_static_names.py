r"""静态检查：不许有未定义的名字。

为什么要有这一条
----------------
2026-11-13：我改 P2 的快照段落时删掉了 `bs = stream.broadcast_state()`，而下面的
`bus.publish(...)` 还在用 `bs`。所有单元测试都通过——因为它们不跑 P2 的运行时循环。
她"看起来活着"（P1 在正常续租），只有 P2 的日志里有 `NameError: bs`，
而且要到 supervisor 连续 5 次放弃重启之后才会安静下来。

这类错误不需要跑起来才能发现：它就是一个作用域里没定义的名字。
venv 里没有 pyflakes、也没有网络（装不了），所以用标准库的 `symtable` 做一个最小版本：
对每个模块，找出"在函数里被当成全局引用、但模块级也没定义、也不是内建"的名字。

局限（写出来，免得被当成全功能 linter）：
- 不检查属性名、参数个数、类型；
- 星号导入会漏（`from x import *` 让所有名字看起来可能已定义）；
- 动态注入（`globals()["x"] = ...`）会误报，所以有白名单，且白名单只允许为空或写明理由。
"""

from __future__ import annotations

import builtins
import pathlib
import symtable
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCAN_DIRS = ("src", "tools")

# 已知的误报（动态定义的名字）。为空是目标；加进来必须写原因。
WHITELIST: set[tuple[str, str]] = set()


def undefined_names(path: pathlib.Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    top = symtable.symtable(src, str(path), "exec")
    module_names = set(top.get_identifiers())
    builtin_names = set(dir(builtins))
    out: list[str] = []

    def walk(table: symtable.SymbolTable) -> None:
        for sym in table.get_symbols():
            if sym.is_global() and not sym.is_assigned():
                name = sym.get_name()
                if name in module_names or name in builtin_names:
                    continue
                out.append(name)
        for child in table.get_children():
            walk(child)

    walk(top)
    return sorted(set(out))


def test_no_undefined_names_in_src_or_tools() -> None:
    offenders: dict[str, list[str]] = {}
    scanned = 0
    for d in SCAN_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            if path.stat().st_size == 0:
                continue
            scanned += 1
            names = [n for n in undefined_names(path) if (path.name, n) not in WHITELIST]
            if names:
                offenders[str(path.relative_to(ROOT))] = names
    assert not offenders, f"undefined names: {offenders}"
    print(f"       {scanned} modules, no undefined names")


def test_the_checker_actually_detects_one() -> None:
    """A checker that never fires is decoration. Feed it the bug it was built for."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "bad.py"
        p.write_text(
            "def f(stream):\n"
            "    return bs['x']\n",
            encoding="utf-8",
        )
        assert undefined_names(p) == ["bs"], undefined_names(p)
    print("       the checker reports the 'bs' bug it was written for")


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
