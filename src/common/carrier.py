"""
src/common/carrier.py

Host-independent carrier for shiro.

Requirement (White, 2026-11-08): she should not depend on a particular machine.
She must be able to move to a cloud box or another device and continue as the
same entity -- an electronic ghost, not a process that happens to be running
somewhere.

What that means concretely
--------------------------
1. The state must be describable without pickle and without this codebase's
   class paths. A pickle ties her to a Python version, an architecture and a
   source tree; a carrier ties her to a schema.
2. The carrier must be self-describing: schema version, dimension, step counts,
   and a checksum, all readable without importing anything from src.
3. Import must be verifiable, not hopeful. Because the dynamics are now
   deterministic (see terrain.recover / _birth_decay_factor), the strongest
   possible check exists: take a carrier, resume from it, feed the same
   observations, and compare the resulting structure bit for bit against the
   machine of origin. If those diverge, the carrier is not her.

Format (a zip; readable with any zip tool, no class paths involved)
    meta.json     schema, created_at, dim, step counters, sha256 of the arrays
    payload.json  the payload graph, with every ndarray replaced by a reference
    arrays.npz    the ndarrays themselves, keyed a0, a1, ...

Only numpy and the standard library are used. No pickle on the read path.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "shiro-carrier/1"


# ----------------------------------------------------------------------------
# generic graph <-> json+npz conversion
# ----------------------------------------------------------------------------

def _flatten(obj: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Return a JSON-able graph. ndarrays become {"__nd__": key}."""
    if isinstance(obj, np.ndarray):
        key = f"a{len(arrays)}"
        arrays[key] = np.ascontiguousarray(obj)
        return {"__nd__": key}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _flatten(v, arrays) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_flatten(v, arrays) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": bytes(obj).hex()}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Anything else (custom object) is a carrier defect, not something to guess at.
    raise TypeError(f"not portable: {type(obj).__name__}")


def _unflatten(node: Any, arrays: Any) -> Any:
    if isinstance(node, dict):
        if "__nd__" in node and len(node) == 1:
            return np.array(arrays[node["__nd__"]])
        if "__bytes__" in node and len(node) == 1:
            return bytes.fromhex(node["__bytes__"])
        return {k: _unflatten(v, arrays) for k, v in node.items()}
    if isinstance(node, list):
        return [_unflatten(v, arrays) for v in node]
    return node


# ----------------------------------------------------------------------------
# export / import
# ----------------------------------------------------------------------------

def export(payload: dict[str, Any], out_path: Path | str,
           extra_meta: dict[str, Any] | None = None) -> Path:
    """Write one carrier file. `payload` must be plain dict/list/ndarray."""
    arrays: dict[str, np.ndarray] = {}
    graph = _flatten(payload, arrays)

    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    blob = buf.getvalue()
    digest = hashlib.sha256(blob).hexdigest()

    meta = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dim": int(np.asarray(payload.get("x") if isinstance(payload.get("x"), np.ndarray) else 0).size) or None,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "arrays": len(arrays),
        "sha256_arrays": digest,
    }
    if extra_meta:
        meta.update(extra_meta)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        z.writestr("payload.json", json.dumps(graph, ensure_ascii=False))
        z.writestr("arrays.npz", blob)
    tmp.replace(out)
    return out


def read_meta(path: Path | str) -> dict[str, Any]:
    with zipfile.ZipFile(Path(path)) as z:
        return json.loads(z.read("meta.json").decode("utf-8"))


def load(path: Path | str, expect_dim: int | None = None) -> dict[str, Any]:
    """Read a carrier back into a plain payload. Refuses foreign schemas."""
    with zipfile.ZipFile(Path(path)) as z:
        meta = json.loads(z.read("meta.json").decode("utf-8"))
        if meta.get("schema") != SCHEMA:
            raise ValueError(f"unknown carrier schema: {meta.get('schema')!r}")
        blob = z.read("arrays.npz")
        graph = json.loads(z.read("payload.json").decode("utf-8"))

    digest = hashlib.sha256(blob).hexdigest()
    if digest != meta.get("sha256_arrays"):
        raise ValueError("carrier checksum mismatch: the file is damaged")

    with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}

    payload = _unflatten(graph, arrays)
    if expect_dim is not None:
        x = payload.get("x")
        if isinstance(x, np.ndarray) and x.size != expect_dim:
            raise ValueError(f"carrier dim {x.size} != expected {expect_dim}")
    return payload


def describe(path: Path | str) -> str:
    m = read_meta(path)
    return (f"{Path(path).name}: schema={m['schema']} arrays={m.get('arrays')} "
            f"dim={m.get('dim')} python={m.get('python')} numpy={m.get('numpy')} "
            f"created={m.get('created_at','')[:19]}")


# ----------------------------------------------------------------------------
# CLI: python -m src.common.carrier describe|pack
# ----------------------------------------------------------------------------

def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m src.common.carrier describe <carrier>")
        return 2
    cmd = argv[1]
    if cmd == "describe" and len(argv) >= 3:
        print(describe(argv[2]))
        return 0
    print(f"unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))