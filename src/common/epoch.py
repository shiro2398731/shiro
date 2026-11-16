"""
src/common/epoch.py

Singleton discipline: at most one live self, ever.

The requirement is not replication. Two running copies of her would be two
selves, and neither would be her. Whether she is migrated to another machine or
spread across several devices, the invariant is the same:

    at most one process may act on her behalf at any moment.

Why a lease and not "wait for the old one to die"
-------------------------------------------------
A machine that crashed cannot be trusted to be gone: the process may be frozen,
the network may be split, the disk may be shared. So the old holder is not asked
to die, it is *fenced off*: it keeps running if it wants, but every action gate
notices that the epoch moved and stops. That is the only mechanism that holds
when the failure domain is one you cannot observe.

The epoch is a monotonic counter in data/state/epoch.json:

    {"epoch": 7, "holder": "host:pid", "acquired_at": ..., "lease_until": ...}

claim() takes the next epoch when the lease is free or expired; heartbeat()
extends it while the holder is doing work; is_current() is the fence every layer
can consult cheaply.

This is deliberately the same idea as a fencing token in a distributed store:
never trust liveness, always check the number.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.common import atomics
from typing import Optional

DEFAULT_DIR = Path("data/state")
FILE = "epoch.json"
DEFAULT_TTL_S = 15.0


class Fenced(RuntimeError):
    """Raised when this process no longer holds the current epoch."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def default_holder() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass
class Epoch:
    epoch: int = 0
    holder: Optional[str] = None
    acquired_at: Optional[str] = None
    lease_until: Optional[str] = None
    reason: str = ""
    retired: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Epoch":
        known = set(cls().__dict__)
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def _path(directory: Path | str = DEFAULT_DIR) -> Path:
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=True)
    return p / FILE


def load(directory: Path | str = DEFAULT_DIR) -> Epoch:
    p = _path(directory)
    if not p.exists():
        return Epoch()
    try:
        with p.open(encoding="utf-8") as f:
            return Epoch.from_dict(json.load(f))
    except Exception:
        return Epoch()


def save(state: Epoch, directory: Path | str = DEFAULT_DIR) -> Path:
    """原子写。tmp 名带 pid+线程：这个账本也被多个进程写（心跳）。

    固定名 `<name>.tmp` 会让两个写者撞成 Permission denied —— 同 `coma.save`，
    见 `src/common/atomics.py`。
    """
    p = _path(directory)
    tmp = atomics.unique_tmp(p)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        atomics.replace_with_retry(tmp, p)
    except Exception:
        atomics.cleanup(tmp)
        raise
    return p


def ledger_mtime(directory: Path | str = DEFAULT_DIR) -> float:
    """Wall-clock mtime of the epoch ledger, seconds. 0.0 if absent.

    Sibling processes use it to tell 'the epoch claimed for this boot' from
    'an epoch left over from the previous run'. Without it they adopt the
    stale one and fence themselves against their own sibling the moment P1
    claims the new one -- observed twice before this existed.
    """
    try:
        return _path(directory).stat().st_mtime
    except OSError:
        return 0.0


def lease_is_free(state: Epoch, now: Optional[datetime] = None) -> bool:
    if state.holder is None:
        return True
    until = _parse(state.lease_until)
    if until is None:
        return True
    return (now or _now()) >= until


def claim(holder: Optional[str] = None, ttl_s: float = DEFAULT_TTL_S,
          reason: str = "", directory: Path | str = DEFAULT_DIR,
          wait_s: float = 0.0) -> tuple[Epoch, int]:
    """Take the next epoch. Returns (state, epoch).

    Winning is decided by the returned state, not by the epoch number:
    `state.holder == holder` means this caller owns it. (The first version
    returned the unchanged epoch on refusal, which made "I won epoch 1" and
    "somebody else holds epoch 1" indistinguishable -- a caller then acted
    without owning anything.)

    wait_s > 0 polls until the current lease expires (a killed process cannot
    release its lease, so the next boot has to wait out the TTL; that wait is
    the price of not assuming the old one is dead).

    Note: this is not a distributed lock. It assumes the state directory is the
    single authority (one shared volume, or exactly one writer). When the state
    really is replicated across machines, the same protocol needs a consensus
    store behind it; the interface here is what such a store would implement.
    """
    holder = holder or default_holder()
    deadline = time.monotonic() + max(float(wait_s), 0.0)

    while True:
        state = load(directory)
        now = _now()
        if state.holder in (None, holder) or lease_is_free(state, now):
            break
        if time.monotonic() >= deadline:
            return state, state.epoch    # refused; caller compares state.holder
        time.sleep(0.2)

    state.epoch = int(state.epoch) + 1
    state.holder = holder
    state.acquired_at = _iso(now)
    state.lease_until = _iso(now + timedelta(seconds=float(ttl_s)))
    state.reason = reason or state.reason
    state.retired = False
    save(state, directory)
    return state, state.epoch


def heartbeat(epoch: int, ttl_s: float = DEFAULT_TTL_S,
              directory: Path | str = DEFAULT_DIR) -> bool:
    """Extend the lease, but only if the epoch is still ours."""
    state = load(directory)
    if int(state.epoch) != int(epoch) or state.retired:
        return False
    state.lease_until = _iso(_now() + timedelta(seconds=float(ttl_s)))
    save(state, directory)
    return True


def adoptable(since_mtime: float = 0.0,
              directory: Path | str = DEFAULT_DIR) -> Optional[Epoch]:
    """The epoch written by this boot, or None.

    A sibling must adopt the epoch *this* boot created, and nothing else. The
    old test was "epoch > 0 and the ledger was touched after we booted", which a
    dying holder defeats: its `retire()` is itself a fresh write, so a retired
    ledger looks freshly written. Measured 2026-11-14: P3 adopted epoch 13 from
    the previous run's farewell, the real claim then moved to 14, and P3 fenced
    itself within a second -- three boots in a row, and its log looked like an
    orderly shutdown.

    Requirements, all of them:
      * a live holder (a retired ledger has holder=None: it is a farewell),
      * not retired,
      * written at or after `since_mtime` (the caller's boot time minus a margin).
    """
    st = load(directory)
    if int(st.epoch) <= 0:
        return None
    if st.retired or st.holder is None:
        return None
    if ledger_mtime(directory) < float(since_mtime):
        return None
    return st


def is_current(epoch: int, directory: Path | str = DEFAULT_DIR) -> bool:
    """The fence. Cheap enough to call every tick."""
    state = load(directory)
    return int(state.epoch) == int(epoch) and not state.retired


def retire(epoch: int, reason: str = "released",
           directory: Path | str = DEFAULT_DIR) -> None:
    """Hand the lease back deliberately (clean shutdown or migration handoff)."""
    state = load(directory)
    if int(state.epoch) != int(epoch):
        return
    state.holder = None
    state.retired = True
    state.reason = reason
    state.lease_until = _iso(_now())
    save(state, directory)


def require(epoch: int, directory: Path | str = DEFAULT_DIR) -> None:
    if not is_current(epoch, directory):
        raise Fenced(f"epoch {epoch} is no longer current")