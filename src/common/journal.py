"""
src/common/journal.py

Append-only journal for L5, plus deterministic replay.

Why this exists
---------------
Before this, L5 was only a checkpoint every 60 s. A kill therefore restored
her to a state from up to 60 s earlier: the time in between was lived and then
discarded. That is a rollback, not a coma. A checkpoint is a compression of the
past; the journal is what happened after it.

The pattern is the one used by durable-execution systems (Temporal, DBOS,
Azure Durable Functions): persist the history, replay it deterministically,
and keep snapshots only as a shortcut so replay does not start from year zero.

Record format (little endian, fixed size per kind):
    b'A' + 128 * float32   one observation handed to StateSpace.record()
    b'R' + uint64          a recovery call, carrying total_steps at that moment

A record is either complete or absent: a torn tail (kill during write) leaves
a short read, which replay simply stops at. That is the whole recovery policy.

Determinism requirement
-----------------------
Replay only reproduces the original state because terrain.py derived both of
its random draws from the step index (see recover() and _birth_decay_factor).
Without that, replaying the same history would produce a different structure
and this file would be a lie.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..L5_cognition.terrain import StateSpace

OBS = b"A"
REC = b"R"
_DIM = 128
_OBS_BYTES = _DIM * 8                      # float64 payload: exact replay, no drift
_DIM_MARK = b"#L5J2"                       # file header, so a foreign file is refused


class Journal:
    """Append observations and recovery calls; replay them into a StateSpace."""

    def __init__(self, path: Path | str = "data/journal/l5.wal", dim: int = _DIM) -> None:
        self.path = Path(path)
        self.dim = dim
        self._obs_bytes = dim * 8
        self._fh: Any = None
        self.records_written = 0
        self._ensure_header()

    def _ensure_header(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("wb") as f:
                f.write(_DIM_MARK)
                f.write(struct.pack("<I", self.dim))
                f.flush()

    def open(self) -> None:
        if self._fh is None:
            self._fh = self.path.open("ab")

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
                self._fh = None

    # ---------- append ----------

    def append_observe(self, x: np.ndarray) -> None:
        self.open()
        arr = np.asarray(x, dtype="<f8").reshape(-1)
        if arr.size != self.dim:
            raise ValueError(f"expected {self.dim} dims, got {arr.size}")
        self._fh.write(OBS)
        self._fh.write(arr.tobytes())
        self.records_written += 1

    def append_recover(self, total_steps: int) -> None:
        self.open()
        self._fh.write(REC)
        self._fh.write(struct.pack("<Q", int(total_steps)))
        self.records_written += 1

    def flush(self, fsync: bool = False) -> None:
        if self._fh is not None:
            self._fh.flush()
            if fsync:
                import os as _os
                _os.fsync(self._fh.fileno())

    # ---------- replay ----------

    def read_entries(self) -> tuple[list[np.ndarray], list[int]]:
        """Read every complete record. A torn tail is dropped, not repaired."""
        observations: list[np.ndarray] = []
        recovers: list[int] = []
        if not self.path.exists():
            return observations, recovers
        with self.path.open("rb") as f:
            head = f.read(len(_DIM_MARK))
            if head != _DIM_MARK:
                return observations, recovers
            raw_dim = f.read(4)
            if len(raw_dim) < 4:
                return observations, recovers
            dim = struct.unpack("<I", raw_dim)[0]
            if dim != self.dim:
                raise ValueError(f"journal dim {dim} != {self.dim}")
            obs_bytes = dim * 8
            while True:
                kind = f.read(1)
                if not kind:
                    break
                if kind == OBS:
                    buf = f.read(obs_bytes)
                    if len(buf) < obs_bytes:
                        break                      # torn tail
                    observations.append(np.frombuffer(buf, dtype="<f8").astype(float))
                elif kind == REC:
                    buf = f.read(8)
                    if len(buf) < 8:
                        break
                    recovers.append(struct.unpack("<Q", buf)[0])
                else:
                    break                          # foreign byte: stop, do not guess
        return observations, recovers

    def replay(self, space: "StateSpace") -> dict[str, int]:
        """Apply the journal to `space` in order. Returns counts."""
        observations, recovers = self.read_entries()
        # Interleaving matters only through total_steps, which record() advances
        # and recover() reads for its seed. Both lists are positionally aligned
        # because the writer appended in the same order the caller produced them.
        # To keep that alignment we re-read the stream once, in order.
        applied_obs = applied_rec = 0
        with self.path.open("rb") as f:
            f.read(len(_DIM_MARK) + 4)
            obs_bytes = self.dim * 8
            while True:
                kind = f.read(1)
                if not kind:
                    break
                if kind == OBS:
                    buf = f.read(obs_bytes)
                    if len(buf) < obs_bytes:
                        break
                    space.record(np.frombuffer(buf, dtype="<f8").astype(float))
                    applied_obs += 1
                elif kind == REC:
                    buf = f.read(8)
                    if len(buf) < 8:
                        break
                    space.recover()
                    applied_rec += 1
                else:
                    break
        return {"observations": applied_obs, "recovers": applied_rec,
                "available": len(observations) + len(recovers)}

    # ---------- compaction ----------

    def truncate(self) -> None:
        """Drop the journal after a fresh checkpoint has captured its effect."""
        self.close()
        with self.path.open("wb") as f:
            f.write(_DIM_MARK)
            f.write(struct.pack("<I", self.dim))
        self.records_written = 0

    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0