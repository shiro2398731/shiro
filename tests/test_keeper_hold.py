r"""keeper 暂停开关的测试（2026-11-28）。

事故：全量测试占满 CPU → 心跳 stale → keeper 连续重启 P1/P2/P3，epoch 被顶掉。
修复：测试/维护时放 `data/state/keeper.hold`，keeper 只记录、不重启。

Run: .\.venv\Scripts\python.exe tests\test_keeper_hold.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import keeper  # noqa: E402


def test_held_reads_file() -> None:
    old_root, old_hold = keeper.ROOT, keeper.HOLD_FILE
    with tempfile.TemporaryDirectory() as d:
        keeper.ROOT = pathlib.Path(d)
        keeper.HOLD_FILE = keeper.ROOT / "data" / "state" / "keeper.hold"
        keeper.HOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
        assert keeper._held() is False
        keeper.HOLD_FILE.write_text("test", encoding="utf-8")
        assert keeper._held() is True
    keeper.ROOT, keeper.HOLD_FILE = old_root, old_hold
    print("       hold file toggles _held()")


def test_hold_prevents_restart() -> None:
    started = []
    old_held, old_start = keeper._held, keeper.start_module
    try:
        keeper._held = lambda: True
        keeper.start_module = lambda tag: started.append(tag) or 12345
        w = keeper.Watcher(stale_s=30.0, min_life_s=60.0)
        rc = keeper._watch(["P1", "P2", "P3"], now=10_000.0, w=w)
        assert rc == 0
        assert started == [], started
    finally:
        keeper._held, keeper.start_module = old_held, old_start
    print("       hold -> no restarts")


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
