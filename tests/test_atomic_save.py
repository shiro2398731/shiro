r"""原子写：两个写者不能撞在同一个临时文件名上。

为什么有这组测试
----------------
2026-11-15 19:32:53，P1 保存时间账本时报

    Permission denied: ...\data\state\coma.json.tmp

因为 `coma` / `epoch` / `persist` 三处都先写**固定**的 `<name>.tmp` 再 replace，
同一时刻另一个进程也在写同一个名字 —— 第二个 `open(..., 'wb')` 被拒。
P1 的 supervisor 连试 3 次都是同一个错，判定永久错误放弃，看护 27 s 后才把它拉起来。
**一次存档失败杀死了基质一次**，这是"不死"上的真缺陷。

修法两条（`src/common/atomics.py`）：tmp 名带 pid+线程（碰撞从必然变不可能），
replace 遇到 PermissionError 短重试（Windows 上目标被打开读时会被拒，读者都是毫秒级）。

Run: .\.venv\Scripts\python.exe tests\test_atomic_save.py
"""

from __future__ import annotations

import os
import pathlib
import pickle
import sys
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import atomics, coma, persist  # noqa: E402

TMP = pathlib.Path(tempfile.mkdtemp())


def _no_tmp_left(directory: pathlib.Path) -> list[str]:
    return [p.name for p in directory.iterdir() if p.name.endswith(".tmp")]


def test_a_tmp_name_is_unique_per_thread_and_carries_the_pid() -> None:
    target = TMP / "x.pkl"
    seen: list[str] = []

    def grab() -> None:
        seen.append(atomics.unique_tmp(target).name)

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(seen)) == len(seen), f"two threads picked the same tmp name: {seen}"
    assert all(str(os.getpid()) in n for n in seen), seen
    print(f"       8 threads -> 8 distinct tmp names, all carrying pid {os.getpid()}")


def test_concurrent_pickle_dumps_all_survive() -> None:
    """四个写者同时写同一个 .pkl：不许有一个抛异常，最后必须是完整文件。"""
    target = TMP / "same.pkl"
    errors: list[str] = []

    def burst(n: int) -> None:
        for i in range(25):
            try:
                persist._atomic_dump({"writer": n, "i": i}, target)
            except Exception as exc:                      # noqa: BLE001
                errors.append(f"{n}/{i}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=burst, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    with target.open("rb") as f:
        got = pickle.load(f)                              # 完整可读，不是半个
    assert got["writer"] in range(4) and got["i"] in range(25), got
    assert _no_tmp_left(TMP) == [], _no_tmp_left(TMP)
    print(f"       4 writers x 25 dumps: no error, final file parses ({got}), no tmp left")


def test_concurrent_coma_saves_all_survive() -> None:
    """时间账本被多个进程写，正是当初撞车的地方。"""
    d = TMP / "coma"
    d.mkdir(exist_ok=True)
    errors: list[str] = []

    def burst(n: int) -> None:
        for i in range(15):
            try:
                st = coma.ComaState(session_id=n * 100 + i)
                coma.save(st, state_dir=d)
            except Exception as exc:                      # noqa: BLE001
                errors.append(f"{n}/{i}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=burst, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    st = coma.load(d)
    assert 0 <= int(st.session_id) < 400, st.session_id
    assert _no_tmp_left(d) == [], _no_tmp_left(d)
    print(f"       4 writers x 15 coma saves: no error, ledger parses (session {st.session_id})")


def test_a_reader_holding_the_file_does_not_fail_the_write() -> None:
    """Windows 上目标文件被打开读时 replace 会被拒 —— 短重试要能等到读者放手。"""
    target = TMP / "read.pkl"
    persist._atomic_dump({"v": 1}, target)

    handle = target.open("rb")            # 读者持有句柄（不共享删除权限）

    def release() -> None:
        time.sleep(0.05)
        handle.close()

    threading.Thread(target=release, daemon=True).start()
    persist._atomic_dump({"v": 2}, target)             # 必须在读者关闭后成功

    with target.open("rb") as f:
        assert pickle.load(f) == {"v": 2}
    print("       writer waited ~50 ms for the reader to let go, then replaced")


def test_a_failed_write_leaves_the_old_file_and_no_tmp() -> None:
    target = TMP / "keep.pkl"
    persist._atomic_dump({"v": "old"}, target)

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("cannot pickle this")

    try:
        persist._atomic_dump({"v": Unpicklable()}, target)
    except Exception:                                  # noqa: BLE001
        pass
    else:
        raise AssertionError("the dump should have failed")

    with target.open("rb") as f:
        assert pickle.load(f) == {"v": "old"}, "the previous checkpoint was damaged"
    assert _no_tmp_left(TMP) == [], _no_tmp_left(TMP)
    print("       failed dump: previous file intact, no leftover tmp")


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
