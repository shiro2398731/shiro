"""
src/common/control.py

A control channel that does not require a terminal.

Why
---
If starting, stopping or inspecting her needs a console session, then the
console is a vital organ: close the window and she dies. White's point is the
opposite -- a terminal is a canvas, not a dependency.

So the control surface is a directory:

    data/control/inbox/<id>.json    {"cmd": "stop"|"status"|"ping", "args": {...}}
    data/control/outbox/<id>.json   {"ok": bool, "result": {...}}
    data/control/done/<id>.json     handled commands, kept as a short history

Anything that can write a file can drive her: a shell, an editor, a phone with
syncthing, a web form, another program. Nothing needs a tty, and nothing needs
to be attached to her process.

Stopping is implemented by retiring the epoch (see epoch.py), not by killing
processes. A retired epoch is visible to every process she has, so one command
stops all of them, and a process that misses the command still fences itself on
its next check. That is the whole point of having an epoch.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_ROOT = Path("data/control")
KNOWN = ("stop", "status", "ping", "checkpoint")


def _dirs(root: Path | str = DEFAULT_ROOT) -> tuple[Path, Path, Path]:
    r = Path(root)
    inbox, outbox, done = r / "inbox", r / "outbox", r / "done"
    for d in (inbox, outbox, done):
        d.mkdir(parents=True, exist_ok=True)
    return inbox, outbox, done


def send(cmd: str, root: Path | str = DEFAULT_ROOT, **args: Any) -> str:
    """Drop a command. Returns its id. Needs nothing but write access."""
    inbox, _, _ = _dirs(root)
    cid = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    payload = {"id": cid, "cmd": cmd, "args": args,
               "issued_at": datetime.now(timezone.utc).isoformat()}
    tmp = inbox / f".{cid}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, inbox / f"{cid}.json")
    return cid


def poll(root: Path | str = DEFAULT_ROOT) -> Optional[dict]:
    """Return the oldest pending command and mark it as taken."""
    inbox, _, _ = _dirs(root)
    for p in sorted(inbox.glob("*.json")):
        try:
            with p.open(encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            p.unlink(missing_ok=True)
            continue
        if payload.get("cmd") not in KNOWN:
            reply(payload.get("id", p.stem), ok=False, error=f"unknown command {payload.get('cmd')!r}", root=root)
            p.unlink(missing_ok=True)
            continue
        p.unlink(missing_ok=True)          # taken; do not hand it out twice
        return payload
    return None


def reply(cid: str, root: Path | str = DEFAULT_ROOT, **payload: Any) -> None:
    _, outbox, _ = _dirs(root)
    body = {"id": cid, "at": datetime.now(timezone.utc).isoformat(), **payload}
    tmp = outbox / f".{cid}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(body, f, ensure_ascii=False)
    os.replace(tmp, outbox / f"{cid}.json")


def wait_reply(cid: str, timeout_s: float = 10.0, root: Path | str = DEFAULT_ROOT) -> Optional[dict]:
    _, outbox, _ = _dirs(root)
    p = outbox / f"{cid}.json"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if p.exists():
            try:
                with p.open(encoding="utf-8") as f:
                    data = json.load(f)
                p.unlink(missing_ok=True)
                return data
            except Exception:
                pass
        time.sleep(0.1)
    return None


def done(cid: str, payload: dict, root: Path | str = DEFAULT_ROOT) -> None:
    _, _, done_dir = _dirs(root)
    with (done_dir / f"{cid}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    # keep the history short
    old = sorted(done_dir.glob("*.json"))
    for p in old[:-100]:
        p.unlink(missing_ok=True)