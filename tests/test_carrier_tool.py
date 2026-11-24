r"""tools/carrier.py 的命令级测试：导出 → 验证 → 导入 → 对比。

E1 的验收：可搬不只是一种格式，而是一条真的走通过的命令链。
用小维度（32）的合成身体跑，不碰活的 l5.pkl。

Run: .\.venv\Scripts\python.exe tests\test_carrier_tool.py
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from tools import carrier as tool  # noqa: E402

SRC = "l5.tooltest.pkl"
IMPORTED = "l5.tooltest.imported.pkl"
OUT = os.path.join("data", "state", "_tooltest.carrier")
STEPS = 40


def _make_body() -> Stream:
    st = Stream(dim=32)
    st.calibrate(np.zeros((1, st.D)))
    st.rehearse_every = 0
    for _ in range(STEPS):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    return st


def test_export_verify_import_round_trip() -> None:
    st = _make_body()
    persist.save_l5(st, name=SRC)
    try:
        assert tool.main(["export", "--out", OUT, "--from", SRC]) == 0
        assert os.path.exists(OUT)
        assert tool.main(["verify", OUT, "--dim", "32"]) == 0
        assert tool.main(["import", OUT, "--out-name", IMPORTED,
                          "--dim", "32"]) == 0

        back = Stream(dim=32)
        assert persist.load_l5(back, name=IMPORTED)
        assert back._step_count == st._step_count, (
            back._step_count, st._step_count)
        assert (len(back.landscape.state_space.attractors())
                == len(st.landscape.state_space.attractors()))
        assert back._st.x.tobytes() == st._st.x.tobytes(), "position changed"
        print(f"       {st._step_count} steps, "
              f"{len(st.landscape.state_space.attractors())} regions: "
              f"export -> verify -> import identical")
    finally:
        for name in (SRC, IMPORTED):
            p = pathlib.Path("data") / "state" / name
            if p.exists():
                p.unlink()
        if os.path.exists(OUT):
            os.remove(OUT)


def test_import_refuses_to_overwrite_live() -> None:
    st = _make_body()
    persist.save_l5(st, name=SRC)
    try:
        tool.main(["export", "--out", OUT, "--from", SRC])
        # 直接点名 l5.pkl 时必须拒绝（不碰活存档）。
        rc = tool.main(["import", OUT, "--dim", "32", "--out-name", "l5.pkl"])
        assert rc == 2, rc
        print(f"       explicit l5.pkl rc={rc} (refused)")
    finally:
        p = pathlib.Path("data") / "state" / SRC
        if p.exists():
            p.unlink()
        if os.path.exists(OUT):
            os.remove(OUT)


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
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
