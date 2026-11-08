r"""Carrier: she must be the same entity on another machine.

The strongest available check is not "the fields match" but "the same future
comes out". Because the dynamics are deterministic, two hosts that load the
same carrier and receive the same observations must end up bit-identical.

Run: .\.venv\Scripts\python.exe tests\test_carrier.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import time
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)   # persist 用的是 data/state 相对路径

from src.common import carrier, persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def _sig(stream) -> tuple[int, int, float, float]:
    ss = stream.landscape.state_space
    attrs = ss.attractors()
    if not attrs:
        return (0, 0, 0.0, 0.0)
    mus = np.stack([a.mu for a in attrs])
    amps = np.array([a.amp for a in attrs])
    return (len(attrs), len(stream.graph), round(float(mus.sum()), 9), round(float(amps.sum()), 9))


def _advance(stream, xs, every=8) -> None:
    for i, x in enumerate(xs):
        stream.landscape.state_space.record(x)
        if (i + 1) % every == 0:
            stream.landscape.state_space.recover()


def test_roundtrip_preserves_structure() -> None:
    a = Stream()
    assert persist.load_l5(a), "archive not found; this test needs data/state/l5.pkl"
    payload = persist.build_l5_payload(a)

    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "her.carrier"
        t0 = time.perf_counter()
        carrier.export(payload, path)
        t_export = (time.perf_counter() - t0) * 1000
        size_kb = path.stat().st_size / 1024

        t0 = time.perf_counter()
        back = carrier.load(path, expect_dim=a.landscape.state_space.D)
        t_load = (time.perf_counter() - t0) * 1000

        b = Stream()
        persist.apply_l5_payload(b, back)

        assert _sig(a) == _sig(b), f"载体还原不一致:\n  A={_sig(a)}\n  B={_sig(b)}"
        meta = carrier.read_meta(path)
        assert meta["schema"] == carrier.SCHEMA
        print(f"       carrier {size_kb:.0f} KB, export {t_export:.0f} ms, load {t_load:.0f} ms, sig={_sig(b)}")


def test_same_future_after_migration() -> None:
    """Booting from the carrier must not just look the same: it must continue
    the same way."""
    a = Stream()
    persist.load_l5(a)
    payload = persist.build_l5_payload(a)

    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "her.carrier"
        carrier.export(payload, path)
        b = Stream()
        persist.apply_l5_payload(b, carrier.load(path))

    rng = np.random.default_rng(99)
    xs = [rng.normal(size=a.landscape.state_space.D) for _ in range(40)]
    _advance(a, xs)
    _advance(b, xs)

    assert _sig(a) == _sig(b), f"同一份载体走了不同的未来:\n  A={_sig(a)}\n  B={_sig(b)}"
    print(f"       40 steps on both sides -> sig={_sig(a)}")


def test_corrupt_carrier_is_refused() -> None:
    a = Stream()
    persist.load_l5(a)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "her.carrier"
        carrier.export(persist.build_l5_payload(a), path)

        # 重写 zip，把 arrays.npz 里的一个字节翻掉（尾部字节可能落在 zip 目录区，
        # 那样 zipfile 会容忍，测不到校验和）
        with zipfile.ZipFile(path) as z:
            parts = {n: z.read(n) for n in z.namelist()}
        blob = bytearray(parts["arrays.npz"])
        blob[len(blob) // 2] ^= 0xFF
        parts["arrays.npz"] = bytes(blob)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for n, data in parts.items():
                z.writestr(n, data)
        try:
            carrier.load(path)
            raise AssertionError("a damaged carrier was accepted")
        except (ValueError, zipfile.BadZipFile) as e:
            print(f"       refused: {type(e).__name__}")


def test_foreign_schema_is_refused() -> None:
    a = Stream()
    persist.load_l5(a)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "her.carrier"
        carrier.export(persist.build_l5_payload(a), path, extra_meta={"schema": "someone-else/9"})
        try:
            carrier.load(path)
            raise AssertionError("foreign schema accepted")
        except ValueError as e:
            print(f"       refused: {e}")


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