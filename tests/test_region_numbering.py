r"""区域编号只有一套：`_attractors` 的下标。

这个代码库曾经有三套编号（表下标 / salient 子集下标 / 调用方自己发明的），
混用造成过三次"看起来有数据"的错误，最严重的一次是 grounding 的引擎表算出
MI = 1.0240 bits，而按广播 id 复算只有 0.7395 —— 同一批数据，两套编号。

现在对外暴露的编号统一为**表下标**。这组测试钉住四件事：

1. `salient_indices()` 与 `salient()` 是同一批对象；
2. `nearest_attractor()` 返回的是表下标，不是子集下标；
3. 它与 `_nearest_normalized()` 在同一空间里给同一个数；
4. 广播给 L6 的 `attractor_id` 也是表下标，且指向离她最近的那个显著区域。

仍然没解决的一件事（写在 terrain 的注释里）：表下标是**位置性**的，
合并/删除会让所有人平移。那需要给区域一个稳定 uid，是下一件事。
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.chdir(pathlib.Path(__file__).resolve().parent.parent)

from src.L5_cognition.terrain import Attractor, StateSpace  # noqa: E402


def _space(n: int = 6, dim: int = 8) -> StateSpace:
    ss = StateSpace(dim=dim)
    for i in range(n):
        a = Attractor(mu=np.full(dim, 0.5 * i + 0.5), amp=0.2 + 0.1 * i)
        ss._attractors.append(a)
    return ss


def test_salient_indices_are_the_same_objects_as_salient() -> None:
    ss = _space()
    by_obj = ss.salient()
    by_idx = [ss.attractors()[i] for i in ss.salient_indices()]
    assert by_obj == by_idx, (by_obj, by_idx)
    assert len(by_idx) < len(ss.attractors()), "the test needs a real subset"
    print(f"       salient_indices() -> {ss.salient_indices()} (of {len(ss.attractors())})")


def test_nearest_attractor_returns_a_table_index() -> None:
    ss = _space()
    target = 4
    idx, z = ss.nearest_attractor(ss.attractors()[target].mu)
    assert idx == target, (idx, target)
    assert z < 1e-9, z
    print(f"       a state at region {target}'s centroid -> index {idx}, z={z:.2e}")


def test_the_two_nearest_lookups_agree() -> None:
    ss = _space()
    for i in ss.salient_indices():
        x = ss.attractors()[i].mu
        a = ss.nearest_attractor(x)
        z, j = ss._nearest_normalized(x)
        assert a is not None
        assert a[0] == j, (i, a, j)
        assert abs(a[1] - z) < 1e-9, (a[1], z)
    print("       nearest_attractor() and _nearest_normalized() give the same index")


def test_the_id_broadcast_to_l6_is_a_table_index_of_a_salient_region() -> None:
    from src.common import persist
    from src.L5_cognition.stream import Stream

    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive for this test")
    for _ in range(40):
        st.landscape.set_noise_amplitude(0.02)
        st.step()

    aid = st.detector.current_attractor
    assert aid is not None
    space = st.landscape.state_space
    sal = space.salient_indices()
    assert 0 <= aid < len(space.attractors()), aid
    assert aid in sal, "the reported id must be a salient region"

    # ...and it must be the nearest one *among the salient set*
    mus = np.stack([space.attractors()[i].mu for i in sal])
    sig = np.array([max(space.attractors()[i].sigma, 1e-12) for i in sal])
    zs = np.linalg.norm(mus - st._st.x, axis=1) / sig
    assert int(sal[int(np.argmin(zs))]) == aid, (sal[int(np.argmin(zs))], aid)
    print(f"       attractor_id={aid} is a table index and the nearest salient region")


def test_the_two_lookups_differ_in_which_set_they_search() -> None:
    """`nearest_attractor` = nearest **salient**; `_nearest_normalized` = nearest **any**.

    Same numbering, different sets. That difference used to be invisible because
    the salient one also returned subset indices; writing it down here so the
    next person does not read a disagreement as a numbering bug.
    """
    ss = _space()
    ss._attractors[0].amp = 0.01          # far below half the deepest -> not salient
    x = ss.attractors()[0].mu
    assert 0 not in ss.salient_indices()
    near_salient = ss.nearest_attractor(x)
    z, j = ss._nearest_normalized(x)
    assert j == 0, j
    assert near_salient is not None and near_salient[0] != 0, near_salient
    print(f"       nearest salient={near_salient[0]}  nearest any={j} "
          f"(the difference is the set, not the numbering)")


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
