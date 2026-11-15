r"""看护的决策部分：谁该被拉起、谁该被放弃。

为什么要单独测这一层
--------------------
真正起进程的那一半（`start_module`）已经有实测证据：2026-11-15 14:22:23 用
`keeper.py --once --tag P2` 把被 `kill -9` 掉、心跳冻在 `06:18:51` 的 P2 拉了回来
（12 s 后心跳 `06:22:34`）。这里钉住的是**判断**，因为它错了会比不工作更糟：
一拍一杀会把同一个模块起成一片，而永久错误上无限重启会烧 CPU、刷日志。

判据：心跳文件 mtime 比 `stale_s` 旧 = 停了。不看进程表——venv 的 `python.exe` 是跳板，
一个模块两个进程，看进程表会数出"三个模块六个进程"的假象。

Run: .\.venv\Scripts\python.exe tests\test_keeper.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import keeper  # noqa: E402

TMP = pathlib.Path(tempfile.mkdtemp())
keeper.LOGS = TMP                       # 心跳与事件都写到临时目录


def _beat(tag: str, age_s: float) -> None:
    p = keeper.heartbeat_path(tag)
    p.write_text("x", encoding="utf-8")
    t = time.time() - age_s
    os.utime(p, (t, t))


def _fresh(name: str) -> "keeper.Watcher":
    for f in TMP.glob("*"):
        f.unlink()
    return keeper.Watcher(stale_s=30.0, min_life_s=15.0, fail_limit=3)


def test_a_live_module_is_left_alone() -> None:
    w = _fresh("live")
    _beat("P1", 1.0)
    assert w.plan(["P1"], time.time()) == []
    print("       heartbeat 1 s old -> nothing to do")


def test_a_stale_or_missing_module_is_started() -> None:
    w = _fresh("stale")
    _beat("P1", 120.0)                  # 停了
    # P2 没有心跳文件（从没起过）
    plan = w.plan(["P1", "P2"], time.time())
    assert plan == ["P1", "P2"], plan
    print(f"       P1 stale 120 s + P2 never -> {plan}")


def test_a_module_that_just_started_is_not_started_again() -> None:
    """冷启动要几秒才写出第一份心跳。这段窗口里再起一个，就是一拍一杀。"""
    w = _fresh("waiting")
    _beat("P1", 300.0)
    now = time.time()
    assert w.plan(["P1"], now) == ["P1"]
    w.note_started("P1", now)
    assert w.plan(["P1"], now + 2.0) == [], "started again while it was still booting"
    assert w.plan(["P1"], now + 20.0) == ["P1"], "gave up waiting too early"
    print("       t+0 start, t+2 s skip, t+20 s start again")


def test_a_module_that_keeps_dying_is_given_up_on() -> None:
    w = _fresh("giveup")
    _beat("P1", 600.0)
    now = time.time()
    for i in range(1, 4):                # fail_limit = 3
        assert w.should_give_up("P1", now) is None
        w.note_started("P1", now + i * 20.0)
    why = w.should_give_up("P1", now + 3 * 20.0 + 20.0)
    assert why is not None, w.fails
    w.given_up["P1"] = why
    assert w.plan(["P1"], time.time()) == [], "still trying after giving up"
    print(f"       3 failures -> {why}")


def test_surviving_resets_the_failure_count() -> None:
    w = _fresh("reset")
    _beat("P1", 600.0)
    now = time.time()
    w.note_started("P1", now)
    w.note_started("P1", now + 20.0)
    assert w.fails["P1"] == 2
    w.note_alive("P1")
    assert w.fails["P1"] == 0
    assert w.should_give_up("P1", now + 60.0) is None
    print("       2 failed starts, then alive -> counter back to 0")


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
