"""
tools/autostart.py

开机自启（用户级，不需要管理员）。

设计约束：
- 不修改系统服务/注册表 HKLM；只写用户 Startup 文件夹（或 HKCU Run，二选一，取文件更可审计）。
- 幂等：重复 install 覆盖同一文件，不产生多条。
- 启动命令必须等价于 `headless.py start`，且幂等（epoch 存活即退出 2），
  因此注销/重启多次不会产生多实例。
- 默认不安装：只提供 install/uninstall/status，由操作者显式执行。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_NAME = "shiro-autostart.cmd"


def startup_dir() -> Path:
    """用户级 Startup 目录。非 Windows 返回 data/autostart（占位，行为一致）。"""
    appdata = os.environ.get("APPDATA")
    if os.name == "nt" and appdata:
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    return ROOT / "data" / "autostart"


def launcher_path(target_dir: Path | None = None) -> Path:
    return (target_dir or startup_dir()) / LAUNCHER_NAME


def script_body() -> str:
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = ROOT / ".venv" / "bin" / "python"
    return (
        "@echo off\r\n"
        f"rem shiro autostart: her existence must not require a login shell\r\n"
        f"cd /d \"{ROOT}\"\r\n"
        f"\"{py}\" tools\\headless.py start >nul 2>&1\r\n"
    )


def install(target_dir: Path | None = None) -> Path:
    d = target_dir or startup_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = launcher_path(d)
    p.write_text(script_body(), encoding="utf-8", newline="")
    return p


def uninstall(target_dir: Path | None = None) -> bool:
    p = launcher_path(target_dir)
    if p.exists():
        p.unlink()
        return True
    return False


def status(target_dir: Path | None = None) -> dict:
    p = launcher_path(target_dir)
    return {"path": str(p), "installed": p.exists(),
            "body_ok": p.exists() and "headless.py start" in p.read_text(encoding="utf-8", errors="replace")}


def main(argv: list[str]) -> int:
    action = argv[1] if len(argv) > 1 else "status"
    if action == "install":
        print(f"[autostart] installed: {install()}")
        return 0
    if action == "uninstall":
        print(f"[autostart] removed: {uninstall()}")
        return 0
    print(status())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))