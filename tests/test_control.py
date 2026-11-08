r"""The control channel: a directory, not a terminal.

If stopping or inspecting her needs a tty, the tty is a vital organ. These tests
pin down that a file is enough.

Run: .\.venv\Scripts\python.exe tests\test_control.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.common import control  # noqa: E402


def test_send_and_poll_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as d:
        cid = control.send("status", root=d, note="hi")
        cmd = control.poll(root=d)
        assert cmd is not None
        assert cmd["id"] == cid
        assert cmd["cmd"] == "status"
        assert cmd["args"]["note"] == "hi"


def test_command_is_handed_out_once() -> None:
    with tempfile.TemporaryDirectory() as d:
        control.send("ping", root=d)
        assert control.poll(root=d) is not None
        assert control.poll(root=d) is None, "a command must not run twice"


def test_unknown_command_is_rejected_with_a_reply() -> None:
    with tempfile.TemporaryDirectory() as d:
        cid = control.send("explode", root=d)
        assert control.poll(root=d) is None
        reply = control.wait_reply(cid, timeout_s=1.0, root=d)
        assert reply is not None and reply["ok"] is False


def test_reply_and_wait() -> None:
    with tempfile.TemporaryDirectory() as d:
        cid = control.send("status", root=d)
        control.poll(root=d)
        control.reply(cid, root=d, ok=True, result={"epoch": 7})
        reply = control.wait_reply(cid, timeout_s=2.0, root=d)
        assert reply["result"]["epoch"] == 7


def test_wait_reply_times_out_cleanly() -> None:
    with tempfile.TemporaryDirectory() as d:
        assert control.wait_reply("nope", timeout_s=0.3, root=d) is None


def test_done_history_is_bounded() -> None:
    with tempfile.TemporaryDirectory() as d:
        for i in range(120):
            control.done(f"c{i:04d}", {"i": i}, root=d)
        left = list((pathlib.Path(d) / "done").glob("*.json"))
        assert len(left) <= 100, len(left)


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