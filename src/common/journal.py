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
FOLD = b"F"
_DIM = 128
_OBS_BYTES = _DIM * 8                      # float64 payload: exact replay, no drift
_DIM_MARK = b"#L5J2"                       # file header, so a foreign file is refused

# Same magnitude bound the checkpoint uses (persist.SAVE_MAGNITUDE_LIMIT): a
# state this far out is the signature of a diverged run, not a memory.
_SAVE_LIMIT = 1e6


class Journal:
    """Append observations and recovery calls; replay them into a StateSpace.

    Not journaled, on purpose: the self-model (M1) and the grounding table. Both
    live in the L5 payload, so a rollback loses at most one persist interval of
    them. Measured (2026-11-10, 1999 transitions): dropping the last 60
    transitions moves M1's prequential loss from 0.492859 to 0.498129 bits
    (+0.0053), while the difference between two length windows of the same run
    is 0.0259 bits -- five times larger. Journaling them would mean a new record
    type, a WAL version bump and a replay path that feeds a Stream instead of a
    StateSpace, to recover 1% of a metric whose natural variation is 5%.
    Revisit only if that loss becomes decision-critical.
    """

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

    def append_fold(self, ids: list[int]) -> None:
        """A fold event: the ids removed from the attractor table.

        Folding must be in the journal, otherwise replay would rebuild a body
        with regions that were deliberately collapsed away -- a different body
        with the same history.
        """
        self.open()
        payload = struct.pack("<I", len(ids)) + b"".join(struct.pack("<I", int(i)) for i in ids)
        self._fh.write(FOLD)
        self._fh.write(payload)
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

    def _scan_bad(self) -> int:
        """Count observations that must not be replayed (non-finite or absurdly large)."""
        bad = 0
        if not self.path.exists():
            return 0
        with self.path.open("rb") as f:
            if f.read(len(_DIM_MARK)) != _DIM_MARK:
                return 0
            raw_dim = f.read(4)
            if len(raw_dim) < 4:
                return 0
            dim = struct.unpack("<I", raw_dim)[0]
            if dim != self.dim:
                return 0
            obs_bytes = dim * 8
            while True:
                kind = f.read(1)
                if not kind:
                    break
                if kind == OBS:
                    buf = f.read(obs_bytes)
                    if len(buf) < obs_bytes:
                        break
                    x = np.frombuffer(buf, dtype="<f8").astype(float)
                    if not np.isfinite(x).all() or float(np.abs(x).max()) > _SAVE_LIMIT:
                        bad += 1
                elif kind == REC:
                    if len(f.read(8)) < 8:
                        break
                elif kind == FOLD:
                    raw = f.read(4)
                    if len(raw) < 4:
                        break
                    f.read(4 * struct.unpack("<I", raw)[0])
                else:
                    break
        return bad

    def replay(self, space: "StateSpace") -> dict[str, int]:
        """Apply the journal to `space` in order. Returns counts.

        Fail closed: **one unusable record voids the whole log.**

        Measured 2026-11-13: the WAL from the 11-08 runaway was replayed onto a
        healthy archive. The absurd records (~1e306) were filtered, but the
        *moderately* wrong ones from the same diverged run were accepted, and her
        archive ended up with 31 regions that no coherent history produced (338
        instead of 307). A log whose state sequence is already wrong cannot be
        repaired record by record -- replaying the survivors rebuilds a body that
        never existed. So: nothing is applied, and the caller is told.
        """
        bad = self._scan_bad()
        if bad:
            return {"observations": 0, "recovers": 0, "folds": 0,
                    "rejected": int(bad), "aborted": 1,
                    "available": int(bad)}
        observations, recovers = self.read_entries()
        # Interleaving matters only through total_steps, which record() advances
        # and recover() reads for its seed. Both lists are positionally aligned
        # because the writer appended in the same order the caller produced them.
        # To keep that alignment we re-read the stream once, in order.
        applied_obs = applied_rec = applied_fold = 0
        rejected_obs = 0
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
                    x = np.frombuffer(buf, dtype="<f8").astype(float)
                    # Refuse on replay what the checkpoint already refuses to
                    # save. 2026-11-13: a journal left over from the 11-08
                    # runaway (values ~1e306) was replayed into a healthy
                    # archive on every start, and the resulting region made the
                    # next gradient blow up -- P2 then died on "non-finite
                    # observation". The archive itself was fine; the log was
                    # the poison.
                    if not np.isfinite(x).all() or float(np.abs(x).max()) > _SAVE_LIMIT:
                        rejected_obs += 1
                        continue
                    space.record(x)
                    applied_obs += 1
                elif kind == REC:
                    buf = f.read(8)
                    if len(buf) < 8:
                        break
                    space.recover()
                    applied_rec += 1
                elif kind == FOLD:
                    raw = f.read(4)
                    if len(raw) < 4:
                        break
                    n = struct.unpack("<I", raw)[0]
                    ids_raw = f.read(4 * n)
                    if len(ids_raw) < 4 * n:
                        break
                    ids = [struct.unpack("<I", ids_raw[i * 4:(i + 1) * 4])[0] for i in range(n)]
                    space.apply_fold(ids)
                    applied_fold += 1
                else:
                    break
        return {"observations": applied_obs, "recovers": applied_rec,
                "folds": applied_fold, "rejected": rejected_obs,
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