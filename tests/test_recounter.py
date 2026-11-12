r"""复述：用她现在的结构重读过去的观测。

三条性质：

1. **读得回**：日志里写进去多少条观测，就读回来多少条（REC / FOLD 记录要正确跳过）。
2. **读得对**：健康运行产生的观测，必须落回它当时所在的区域（不是"最近的某个区域"）。
3. **不发明**：明显不属于任何区域的观测返回 none，不拿最近的那个凑数。

实测背景（2026-11-12）：仓库里唯一那份 `data/journal/l5.wal`（11-08）解析出 22 条观测，
全部落空 —— 那不是读错了，是那份日志来自**数值爆炸的那次运行**（观测里出现 -6.8e+236，
即文档 §5.9 记的事故）。所以这里的测试用的是当场写出来的健康日志。

Run: .\.venv\Scripts\python.exe tests\test_recounter.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

from src.common.journal import Journal  # noqa: E402
from src.L5_cognition.recounter import read_observations, recounter  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def _run_with_journal(steps: int = 240, dim: int = 32):
    """一段健康运行，附带它自己的日志与它当时所在的区域序列。"""
    d = tempfile.mkdtemp()
    st = Stream(dim=dim)
    st.calibrate(np.zeros((1, dim)))
    j = Journal(pathlib.Path(d) / "t.wal", dim=dim)
    st.landscape.state_space.journal = j
    visited = []
    for _ in range(steps):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
        aid = st.detector.current_attractor
        visited.append(-1 if aid is None else int(aid))
    j.close()
    st.landscape.state_space.journal = None
    return st, pathlib.Path(d) / "t.wal", visited


def test_count_round_trips() -> None:
    st, wal, _visited = _run_with_journal(120)
    X = read_observations(wal)
    assert X.shape == (120, st.D), X.shape
    print(f"       {X.shape[0]} observations read back, dim={X.shape[1]}")


def test_what_her_structure_can_hold_has_no_recency_bias() -> None:
    """复述的命中率不是"越高越好"，它就是她现在的结构能容纳多少过去。

    我先猜"越近的过去越容易被容纳"（结构是为现在长出来的），**实测把这个猜测否掉了**：
    240 步、dim=32，前一半 42.5%、后一半 42.5% —— 一模一样。
    所以断言写成"没有明显的新近偏置"，而不是我原来那个方向。
    整体只有 42.5% 能落回区域，是因为区域后来才长出来、也会漂移与合并。
    """
    st, wal, visited = _run_with_journal(240)
    X = read_observations(wal)
    r = recounter(st, X)
    half = r["n"] // 2
    resolved = [row["region"] is not None for row in r["rows"]]
    early = sum(resolved[:half]) / max(half, 1)
    late = sum(resolved[half:]) / max(r["n"] - half, 1)
    assert abs(late - early) < 0.15, (early, late)
    print(f"       前一半 {early:.1%} / 后一半 {late:.1%}"
          f"（整体 {r['resolved'] / r['n']:.1%}）—— 没有新近偏置")


def test_rereading_mostly_agrees_with_where_she_was() -> None:
    """落得进去的那些，应当多数就是她当时所在的区域（不是"附近的另一个"）。"""
    st, wal, visited = _run_with_journal(240)
    X = read_observations(wal)
    r = recounter(st, X)
    agree = checked = 0
    for row, was in zip(r["rows"], visited):
        if row["region"] is None or was < 0:
            continue
        checked += 1
        agree += int(row["region"] == was)
    match = agree / checked if checked else 0.0
    assert checked > 10, checked
    assert match >= 0.5, f"agreement only {match:.1%} on {checked} cases"
    print(f"       落得进去的 {checked} 条里，{match:.1%} 与她当时所在区域一致")


def test_a_foreign_state_is_not_forced_into_a_region() -> None:
    st, _wal, _v = _run_with_journal(60)
    far = np.full((1, st.D), 1e6)
    r = recounter(st, far)
    assert r["resolved"] == 0, r
    assert r["rows"][0]["label"] == "none", r["rows"][0]
    print("       a state far outside every region resolves to none")


def test_the_documented_runaway_log_resolves_to_nothing() -> None:
    """The only WAL in the repo is from a diverged run; rereading it must say so.

    This is not a bug in the reader: values around 1e236 are the 11-08 runaway
    (§5.9). If this ever starts resolving, either the log was replaced or the
    guard became too permissive.
    """
    wal = pathlib.Path("data/journal/l5.wal")
    if not wal.exists():
        print("       (no repo WAL, skipped)")
        return
    from src.common import persist

    st = Stream()
    if not persist.load_l5(st):
        print("       (no archive, skipped)")
        return
    X = read_observations(wal)
    if X.size == 0:
        print("       (empty WAL, skipped)")
        return
    r = recounter(st, X)
    biggest = float(np.nanmax(np.abs(X)))
    print(f"       runaway log: n={r['n']} resolved={r['resolved']} "
          f"max|x|={biggest:.3g}")
    assert r["resolved"] == 0 or biggest < 1e6, "a runaway log should not resolve"


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
