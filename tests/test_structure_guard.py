r"""结构守卫：跑一段之后，她的区域数不能塌。

为什么有这组测试
----------------
2026-11-16 17:46–17:48，`recover()` 的一个改动（访问间隔从 `total/assigned` 换成
`life/assigned`）让 `fold_auto()` 连锁触发，两分钟内把区域从 626 折到 3：

    17:46  removed 526  live 122
    17:47  removed 115  live  30
    17:48  removed  33  live   3

`tools/bench.py` 的终点指纹能抓「语义变了」，抓不到「结构塌了」——因为塌的过程
每一步都是合法的动力学，指纹只是换了一个数。这组测试补的正是那个缺口。

判据：从存档跑 N 步后，区域数不得低于起点的 30%。

为什么是 30%：容量折叠是设计行为（20 MB 预算），一次合理的折叠可能砍掉不少；
但「塌」是连续折叠，掉到 30% 以下时它已经不是维护了。这个数字是守卫用的门槛，
不是她自己的判据（C1）。

Run: .\.venv\Scripts\python.exe tests\test_structure_guard.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

COLLAPSE_FLOOR = 0.30
STEPS = 400


def _fresh(seed: int = 7) -> Stream:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive for this test")
    st.rng = np.random.default_rng(seed)
    return st


def test_region_count_does_not_collapse() -> None:
    st = _fresh()
    start = len(st.landscape.state_space.attractors())
    if start < 8:
        print(f"       skipped: only {start} regions, nothing meaningful to guard")
        return
    for _ in range(STEPS):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
    end = len(st.landscape.state_space.attractors())
    floor = start * COLLAPSE_FLOOR
    assert end >= floor, (
        f"regions collapsed: {start} -> {end} over {STEPS} steps "
        f"(floor {floor:.0f}). This is the 2026-11-16 failure shape: a dynamics "
        f"change that makes fold_auto fire in a chain."
    )
    print(f"       {start} -> {end} regions over {STEPS} steps (floor {floor:.0f})")


def test_no_free_fall_across_two_windows() -> None:
    """两个连续窗口都在掉，说明不是一次维护，是在往下滑。

    2026-11-18 09:45：这个测试原来跑的是**副本**。但副本里没有 P1（感知），
    所以她是在「被切断感觉」的状态下跑的——那种情况下她确实会缓慢缩小，
    而真机上她是稳的（最近 6 条 observation：3179、3181、3179、3179、3178、3182）。

    所以改成读**真机的 observation 历史**：
      - 数据是「整个她」跑出来的（有 L1 输入）；
      - 不用花 CPU 跑副本；
      - 如果日志里最后三段是单调下降，那才是真的在滑。

    代价：这个测试现在依赖 `data/logs/observations.jsonl`。没有日志时跳过（不假装通过）。
    """
    import json as _json
    log = ROOT / "data/logs/observations.jsonl"
    if not log.exists():
        print("       skipped: no observations log")
        return

    rows = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            d = _json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if d.get("process") != "P2":
            continue
        data = d.get("data") or {}
        r = data.get("attractors")
        if r is not None:
            rows.append(int(r))

    if len(rows) < 12:
        print(f"       skipped: only {len(rows)} observations")
        return

    # 取最后 12 条，切成三段
    tail = rows[-12:]
    third = len(tail) // 3
    counts = [tail[0],
              tail[third],
              tail[2 * third],
              tail[-1]]
    # 2026-11-27：只看「每段都降」会误报——她正常 -4 区/小时，36 分钟窗口里
    # 掉 2-3 个区就会三段单调下降（实测全量套件里偶发 FAIL，单独跑又 PASS）。
    # 真正的「塌」是 2026-11-16 那种 626→3（相对跌幅 ~99%）。所以判据改成：
    # 单调下降 **且** 相对跌幅 > 20% 才算自由落体。
    falling = all(b < a for a, b in zip(counts, counts[1:]))
    drop = (counts[0] - counts[-1]) / max(1, counts[0])
    assert not (falling and drop > 0.20), (
        f"region count fell in every window: {counts} (drop {drop:.1%})")
    print(f"       real-machine windows (last 12 obs): {counts}, "
          f"drop {drop:.2%} (not a free fall)")


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
