r"""Autostart must be idempotent and must not depend on a login shell.

Her existence must not require someone to open a terminal. The launcher is a
file in the user Startup folder whose only job is to run `headless.py start`,
which is itself idempotent (a live epoch makes it exit 2), so repeated logins
cannot produce more than one instance.

Run: .\.venv\Scripts\python.exe tests\test_autostart.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools import autostart  # noqa: E402


def test_install_writes_a_launcher_pointing_at_this_tree() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = autostart.install(pathlib.Path(d))
        body = p.read_text(encoding="utf-8")
        assert "headless.py start" in body
        assert str(autostart.ROOT) in body, "launcher must cd into this tree"
        assert "python" in body.lower()


def test_install_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as d:
        target = pathlib.Path(d)
        autostart.install(target)
        autostart.install(target)
        files = list(target.iterdir())
        assert len(files) == 1, files
        assert autostart.status(target)["installed"] is True


def test_uninstall_removes_it() -> None:
    with tempfile.TemporaryDirectory() as d:
        target = pathlib.Path(d)
        autostart.install(target)
        assert autostart.uninstall(target) is True
        assert autostart.status(target)["installed"] is False
        assert autostart.uninstall(target) is False


def test_status_reports_body_integrity() -> None:
    with tempfile.TemporaryDirectory() as d:
        target = pathlib.Path(d)
        autostart.install(target)
        st = autostart.status(target)
        assert st["body_ok"] is True
        autostart.launcher_path(target).write_text("echo nothing\r\n", encoding="utf-8")
        assert autostart.status(target)["body_ok"] is False, "a corrupted launcher must be visible"


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