r"""连续性审计工具的测试（2026-11-27）。

钉：gather() 返回四块（coma/epoch/heartbeats/crashes/self_memory）；
她的 L5 状态里没有断点记忆字段（epoch/coma/crash）；心跳数据可读。

Run: .\.venv\Scripts\python.exe tests\test_continuity_audit.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.continuity_audit import gather, self_memory_fields  # noqa: E402


def test_gather_shape() -> None:
    g = gather()
    assert set(g) >= {"coma", "epoch", "heartbeats_s", "crashes", "self_memory"}, g.keys()
    assert isinstance(g["coma"], dict)
    assert isinstance(g["epoch"], dict)
    assert isinstance(g["heartbeats_s"], dict)
    print(f"       epoch={g['epoch'].get('epoch')}, "
          f"coma_events={g['coma'].get('coma_events')}")


def test_no_self_memory_of_breaks() -> None:
    hits = self_memory_fields()
    assert hits == [], hits
    print("       no epoch/coma/crash fields in her state")


def test_heartbeats_present() -> None:
    g = gather()
    ages = g["heartbeats_s"]
    assert "P1" in ages and "P2" in ages and "P3" in ages
    assert ages["P2"] is not None and ages["P2"] < 120.0, ages
    print(f"       heartbeats={ages}")


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
