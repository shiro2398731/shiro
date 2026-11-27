r"""The bundle: one portable body, one present at a time.

The model being tested: a bundle is her *now* (state + journal), not her code and
not her limbs. Standbys exist, but only one may be the present, and a standby
becomes the present deliberately -- never implicitly.

Run: .\.venv\Scripts\python.exe tests\test_bundle.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import bundle, epoch, persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from src.runtime.process_1_fast import FHNNetwork, L2StateField
from src.runtime.prediction import OnlinePredictor  # noqa: E402


def _live() -> dict:
    net, pred, field = FHNNetwork(), OnlinePredictor(), L2StateField()
    persist.restore_all({"l0": net, "l1": pred, "l2": field, "acc": net.accumulator})
    stream = Stream()
    persist.restore_all({"l5": stream})
    return {"net": net, "pred": pred, "field": field, "acc": net.accumulator, "stream": stream}


def _sig(objs: dict) -> tuple:
    ss = objs["stream"].landscape.state_space
    attrs = ss.attractors()
    mus = np.stack([a.mu for a in attrs]) if attrs else np.zeros((0, 1))
    amps = np.array([a.amp for a in attrs]) if attrs else np.zeros(1)
    return (len(attrs), len(objs["stream"].graph), round(float(mus.sum()), 9),
            round(float(amps.sum()), 9), int(objs["net"].step_count),
            round(float(objs["acc"].P), 9))


def test_pack_and_doctor() -> None:
    objs = _live()
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "her.bundle"
        bundle.pack(objs, path)
        assert bundle.doctor(path) == [], bundle.doctor(path)
        meta = bundle.info(path)
        assert meta["kind"] == "bundle"
        assert set(meta["layers"]) == {"l0", "l1", "l2", "acc", "l5"}
        assert meta["engine"] == bundle.ENGINE_VERSION
        print(f"       bundle {path.stat().st_size/1024:.0f} KB, layers={meta['layers']}, "
              f"attractors={meta['attractors']}, journal={meta['journal_bytes']} B")


def test_unpack_reproduces_the_same_body_and_future() -> None:
    src = _live()
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "her.bundle"
        bundle.pack(src, path)
        payload = bundle.load(path)

        dst = {"net": FHNNetwork(), "pred": OnlinePredictor(), "field": L2StateField(),
               "acc": None, "stream": Stream()}
        dst["acc"] = dst["net"].accumulator
        bundle.apply(dst, payload, journal_path=pathlib.Path(d) / "l5.wal")

        assert _sig(src) == _sig(dst), f"本体不一致:\n  src={_sig(src)}\n  dst={_sig(dst)}"

        rng = np.random.default_rng(5)
        xs = [rng.normal(size=src["stream"].landscape.state_space.D) for _ in range(20)]
        for objs in (src, dst):
            for i, x in enumerate(xs):
                objs["stream"].landscape.state_space.record(x)
                if (i + 1) % 8 == 0:
                    objs["stream"].landscape.state_space.recover()

        assert _sig(src) == _sig(dst), f"同一份本体走了不同的未来:\n  src={_sig(src)}\n  dst={_sig(dst)}"
        print(f"       same body and same future: {_sig(dst)}")


def test_doctor_refuses_foreign_engine() -> None:
    objs = _live()
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "future.bundle"
        bundle.pack(objs, path, extra={"engine": "9.9.9"})
        problems = bundle.doctor(path)
        assert any("engine mismatch" in p for p in problems), problems
        print(f"       refused: {problems}")


def test_doctor_refuses_corrupt_payload() -> None:
    objs = _live()
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "her.bundle"
        bundle.pack(objs, path)
        with zipfile.ZipFile(path) as z:
            parts = {n: z.read(n) for n in z.namelist()}
        blob = bytearray(parts["arrays.npz"])
        blob[len(blob) // 2] ^= 0xFF
        parts["arrays.npz"] = bytes(blob)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for n, data in parts.items():
                z.writestr(n, data)
        problems = bundle.doctor(path)
        assert problems and "unreadable" in problems[0], problems
        print(f"       refused: {problems[0][:60]}")


def test_standby_cannot_become_present_while_a_lease_is_live() -> None:
    with tempfile.TemporaryDirectory() as d:
        epoch.claim(holder="present-host:1", ttl_s=30, directory=d)
        st, live = bundle.present(d)
        assert live, "a fresh claim must count as live"
        epoch.retire(st.epoch, directory=d)
        _, live_after = bundle.present(d)
        assert not live_after, "a retired epoch must not block a standby"


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