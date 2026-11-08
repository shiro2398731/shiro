r"""Atomic state writes: a hard kill must not truncate an existing shape.

Run: .\.venv\Scripts\python.exe tests\test_persist_atomic.py
"""

from __future__ import annotations

import pathlib
import pickle
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.common import persist  # noqa: E402


class _Acc:
    HIST = 200

    def __init__(self):
        self.P = 0.42
        self._hist = [0.1] * 10


def test_failed_dump_keeps_previous_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        target = pathlib.Path(d) / "x.pkl"
        persist._atomic_dump({"a": 1}, target)
        before = target.read_bytes()

        original = pickle.dump

        def boom(*_a, **_k):
            raise RuntimeError("simulated crash mid-write")

        pickle.dump = boom
        try:
            persist._atomic_dump({"a": 2}, target)
            raise AssertionError("should have raised")
        except RuntimeError:
            pass
        finally:
            pickle.dump = original

        assert target.read_bytes() == before, "旧文件被破坏了"
        assert not (pathlib.Path(str(target)) .with_suffix(".pkl.tmp")).exists(), "残留了半个 tmp"


def test_normal_roundtrip_still_works() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = persist.STATE_DIR
        persist.STATE_DIR = pathlib.Path(d)
        try:
            acc = _Acc()
            persist.save_acc(acc, "acc.pkl")
            acc2 = _Acc()
            acc2.P = 0.0
            acc2._hist = []
            assert persist.load_acc(acc2, "acc.pkl") is True
            assert abs(acc2.P - 0.42) < 1e-9
            assert len(acc2._hist) == 10
        finally:
            persist.STATE_DIR = old


def test_no_tmp_left_behind_on_success() -> None:
    with tempfile.TemporaryDirectory() as d:
        target = pathlib.Path(d) / "y.pkl"
        persist._atomic_dump({"ok": True}, target)
        leftovers = list(pathlib.Path(d).glob("*.tmp"))
        assert not leftovers, leftovers


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
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())