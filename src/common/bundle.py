"""
src/common/bundle.py

Her body as one portable, extensible, self-contained bundle.

Model (White, 2026-11-08): at any moment exactly one bundle is *the present*;
the others are standbys. A standby may only become the present deliberately,
and two bundles must never run at once -- that is what the epoch fence is for.

Three parts, deliberately kept apart:
    state     what she is now. Cannot be regenerated. Always in the bundle.
    engine    the rules that carry her forward. Replaceable, so the bundle
              records the version it needs instead of embedding the code.
    territory remote limbs. Never in the bundle: a duplicated limb is a second
              self, and there is only ever one.

A bundle is a carrier archive (zip: meta.json + payload.json + arrays.npz)
whose payload holds every layer plus the journal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import carrier, persist

SCHEMA = "shiro-bundle/1"
ENGINE_VERSION = "0.4.0"
REQUIRED_LAYERS = ("l0", "l1", "l2", "acc", "l5")


def engine_compatible(version: str | None) -> bool:
    """Same major.minor is enough; patch versions must not break a body."""
    if not version:
        return False
    cur = ENGINE_VERSION.split(".")[:2]
    got = str(version).split(".")[:2]
    return cur == got


def _journal_bytes(journal_path: Path | str | None) -> bytes | None:
    if not journal_path:
        return None
    p = Path(journal_path)
    return p.read_bytes() if p.exists() else None


def pack(objs: dict[str, Any], out_path: Path | str, *,
         journal_path: Path | str | None = "data/journal/l5.wal",
         epoch_at_pack: int | None = None,
         encoder_id: str = "",
         extra: dict[str, Any] | None = None) -> Path:
    """Write one bundle from live objects: net, pred, field, acc, stream."""
    stream = objs["stream"]
    payload: dict[str, Any] = {
        "l0": persist.build_l0_payload(objs["net"]),
        "l1": persist.build_l1_payload(objs["pred"]),
        "l2": persist.build_l2_payload(objs["field"]),
        "acc": persist.build_acc_payload(objs["acc"]),
        "l5": persist.build_l5_payload(stream),
    }
    journal = _journal_bytes(journal_path)
    if journal:
        payload["journal"] = journal

    ss = stream.landscape.state_space
    meta = {
        "kind": "bundle",
        "engine": ENGINE_VERSION,
        "engine_compat": ".".join(ENGINE_VERSION.split(".")[:2]) + ".x",
        "dim": int(ss.D),
        "layers": sorted(k for k in payload if k != "journal"),
        "journal_bytes": len(journal) if journal else 0,
        "epoch_at_pack": epoch_at_pack,
        "encoder_id": encoder_id,
        "attractors": len(ss.attractors()),
        "edges": len(stream.graph),
        "step_count": int(getattr(stream, "_step_count", 0)),
        "l0_step_count": int(getattr(objs["net"], "step_count", 0)),
    }
    if extra:
        meta.update(extra)
    return carrier.export(payload, out_path, extra_meta=meta)


def doctor(path: Path | str) -> list[str]:
    """Everything that would stop this bundle from being her, as a list.

    Empty list = the machine can carry her. Non-empty = refuse to boot, rather
    than half-start something that looks alive.
    """
    problems: list[str] = []
    p = Path(path)
    if not p.exists():
        return [f"bundle not found: {p}"]

    meta = carrier.read_meta(p)
    if meta.get("schema") != carrier.SCHEMA:
        problems.append(f"unknown carrier schema: {meta.get('schema')!r}")
    if meta.get("kind") != "bundle":
        problems.append("this carrier is not a full bundle (state only)")
    if not engine_compatible(meta.get("engine")):
        problems.append(
            f"engine mismatch: bundle needs {meta.get('engine')!r}, "
            f"this tree is {ENGINE_VERSION}")

    try:
        payload = carrier.load(p)
    except Exception as e:                      # checksum, zip, json
        problems.append(f"payload unreadable: {type(e).__name__}: {e}")
        return problems

    missing = [k for k in REQUIRED_LAYERS if k not in payload]
    if missing:
        problems.append(f"missing layers: {', '.join(missing)}")

    l5 = payload.get("l5") or {}
    try:
        persist.assert_l5_finite(l5)
    except Exception as e:
        problems.append(f"l5 shape is not sound: {e}")

    dim = meta.get("dim")
    x = (l5 or {}).get("x")
    if isinstance(x, np.ndarray) and dim and x.size != int(dim):
        problems.append(f"dim mismatch: payload {x.size}, manifest {dim}")

    n_attr = meta.get("attractors")
    got = len(l5.get("attractors") or [])
    if isinstance(n_attr, int) and n_attr != got:
        problems.append(f"attractor count differs: manifest {n_attr}, payload {got}")

    return problems


def present(path_dir=None):
    """(epoch_state, is_live). "Live" means someone is holding a valid lease.

    A standby bundle must never become the present while this is True --
    that would create a second self rather than moving the one that exists.
    """
    from . import epoch as _epoch
    st = _epoch.load(path_dir) if path_dir else _epoch.load()
    live = (st.holder is not None) and not st.retired and not _epoch.lease_is_free(st)
    return st, live


def load(path: Path | str) -> dict[str, Any]:
    return carrier.load(Path(path))


def apply(objs: dict[str, Any], payload: dict[str, Any],
          journal_path: Path | str | None = "data/journal/l5.wal") -> dict[str, int]:
    """Restore live objects from a bundle payload, and the journal beside it."""
    net, pred, field = objs["net"], objs["pred"], objs["field"]
    acc, stream = objs["acc"], objs["stream"]

    for k, v in (payload.get("l0") or {}).items():
        if k == "envelope_A":
            continue
        if hasattr(net, k):
            setattr(net, k, v)

    for k, v in (payload.get("l1") or {}).items():
        if hasattr(pred, k):
            setattr(pred, k, v)

    l2 = payload.get("l2") or {}
    for i, h in enumerate(l2.get("dim_hist") or []):
        field._dim_hist[i].extend(h)
    # The attribute names are not uniform: only _vec_hist carries the underscore.
    # Copying load_l2's mapping exactly, because a wrong name here silently
    # restores a body with empty normalisation histories.
    field._vec_hist.extend(l2.get("vec_hist") or [])
    for name in ("s_t_hist", "fast_hist", "pressure_hist", "per_fast_mean",
                 "per_amplitude", "per_up_ratio", "per_step_hist"):
        getattr(field, name).extend(l2.get(name) or [])

    accp = payload.get("acc") or {}
    acc.P = accp.get("P", getattr(acc, "P", 0.0))
    acc._hist.extend(accp.get("hist") or [])

    persist.apply_l5_payload(stream, payload.get("l5") or {})

    journal = payload.get("journal")
    written = 0
    if journal and journal_path:
        jp = Path(journal_path)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_bytes(journal)
        written = len(journal)
    return {"layers": len([k for k in payload if k != "journal"]),
            "journal_bytes": written}


def info(path: Path | str) -> dict[str, Any]:
    meta = carrier.read_meta(Path(path))
    problems = doctor(Path(path))
    meta = dict(meta)
    meta["doctor"] = "ok" if not problems else problems
    return meta