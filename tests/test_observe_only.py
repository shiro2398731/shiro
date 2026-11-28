r"""「系统只观察，不调节」的守卫（2026-11-28）。

白默：「系统不替她调节，只能是观察，对所有流程适用。」

这个测试钉住三条：
- Homeostasis 只记录（不调噪声/不改状态/不保存）；
- process_3 收到 state_flat 只写事件；
- ARCHITECTURE 不再写「上调噪声」这类旧说法。

Run: .\.venv\Scripts\python.exe tests\test_observe_only.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_homeostasis_observe_only() -> None:
    src = _read("src/common/homeostasis.py")
    assert "只记录" in src or "只看" in src, "homeostasis 应写明只观察"
    for forbidden in ("set_noise_amplitude", "apply_l5_payload", "save_l5"):
        assert forbidden not in src, f"homeostasis 不应调用 {forbidden}"
    print("       homeostasis is observe-only")


def test_process3_state_flat_only_records() -> None:
    src = _read("src/runtime/process_3_life.py")
    assert 'observer.event("state_flat"' in src, "state_flat 应只写事件"
    assert "set_noise_amplitude" not in src, "process_3 不应调噪声"
    assert "apply_l5_payload" not in src, "process_3 不应改 L5 状态"
    print("       process_3 records state_flat, never adjusts")


def test_architecture_has_no_old_adjust_claim() -> None:
    text = _read("ARCHITECTURE.md")
    assert "上调 L5 噪声" not in text, "ARCHITECTURE 还留着旧的外部调节说法"
    assert "不调噪声" in text or "只记录" in text
    print("       ARCHITECTURE says observe-only")


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
