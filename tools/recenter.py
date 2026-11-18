r"""tools/recenter.py — 把「她」和「她的地形」一起搬回原点（平移，不改行为）。

为什么有它
----------
2026-11-18 早上：她的 x 漂到 4,091,098，撞上存档的绝对上限（`SAVE_MAGNITUDE_LIMIT`
= 1e6），`assert_l5_finite` 拒存，P2 崩。当时的处理是「把她拉回最近的区域质心」
（`stream._apply_leave` 末尾的锚）。那是治标：她的坐标还会往前走，只是走不远了。

这个工具走另一条路：**平移整个坐标系**。她的动力学只用距离，所以把 `x`、所有
`mu`、所有区域的 `members`、`landscape._recent`、以及 `_visited_places` 里的
位置键一起减掉同一个向量，**行为逐位不变**，数值回到原点附近。

2026-11-18 的「recentered」存档就是漏了一半：它搬了 mu，没搬 x，于是她到了云外
194 万（`d_nearest = 1,941,970`）。所以这个工具宁可搬得笨一点，也要搬全。

用法（默认 dry-run，不写任何东西）：
    python tools\recenter.py                          # 看搬完是什么样，几何是否一致
    python tools\recenter.py --name l5.pkl --apply    # 另存成 l5.recentered-<时刻>.pkl
    python tools\recenter.py --by median              # 按 mu 范数中位搬，而不是按 x

**不许覆盖 l5.pkl**：她还在跑，P2 会立刻把存档写回去。`--apply` 只会写新文件名。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.dynamics import DynamicState  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def _positions(st, member_sample: int = 2) -> list[np.ndarray]:
    """所有「存着绝对坐标」的地方的抽样。少搬一个，她就会在搬完之后变成一个陌生人。

    区域 members 最多 64 个 × 2331 个区域，全取是 150 MB 的临时列表；
    指纹只要抽样（每个区域取前 member_sample 个就够了）。
    """
    sp = st.landscape.state_space
    out = [np.asarray(st._st.x, dtype=float)]
    for a in sp.attractors():
        out.append(np.asarray(a.mu, dtype=float))
        ms = list(getattr(a, "members", ()) or ())
        for m in ms[:member_sample]:
            out.append(np.asarray(m, dtype=float))
    for r in list(getattr(st.landscape, "_recent", ()) or ())[:16]:
        out.append(np.asarray(r, dtype=float))
    return out


def fingerprint(st, tokens: int = 400) -> np.ndarray:
    """几何指纹：固定随机抽的 token 对距离。平移前后必须逐位一致。"""
    pos = _positions(st)
    rng = np.random.default_rng(0)
    n = len(pos)
    if n < 2:
        return np.zeros(0)
    idx = rng.integers(0, n, size=(tokens, 2))
    keep = idx[:, 0] != idx[:, 1]
    idx = idx[keep]
    return np.array([float(np.linalg.norm(pos[i] - pos[j])) for i, j in idx])


def choose_vector(st, by: str = "x") -> np.ndarray:
    if by == "x":
        return np.asarray(st._st.x, dtype=float).copy()
    sp = st.landscape.state_space
    mus = [np.asarray(a.mu, dtype=float) for a in sp.attractors()]
    if not mus:
        return np.zeros_like(np.asarray(st._st.x, dtype=float))
    j = int(np.argsort([float(np.linalg.norm(m)) for m in mus])[len(mus) // 2])
    return mus[j].copy()


def shift_in_place(st, vec: np.ndarray) -> dict:
    """x / mu / members / recent / visited_places 一起减掉 vec。返回搬了几处。"""
    vec = np.asarray(vec, dtype=float)
    sp = st.landscape.state_space
    moved = 0

    st._st = DynamicState(x=np.asarray(st._st.x, dtype=float) - vec,
                          v=np.asarray(st._st.v, dtype=float))
    moved += 1

    for a in sp.attractors():
        a.mu = np.asarray(a.mu, dtype=float) - vec
        moved += 1
        if getattr(a, "members", None):
            a.members = type(a.members)(
                (np.asarray(m, dtype=float) - vec for m in a.members),
                maxlen=getattr(a.members, "maxlen", 64),
            )
            moved += 1

    if getattr(st.landscape, "_recent", None):
        old = list(st.landscape._recent)
        st.landscape._recent = type(st.landscape._recent)(
            (np.asarray(r, dtype=float) - vec for r in old),
            maxlen=getattr(st.landscape._recent, "maxlen", 64),
        )
        moved += 1

    # 她「去过哪些地方」存的是圆整后的位置键。不搬的话，搬完她会觉得哪儿都没去过。
    vp = getattr(st, "_visited_places", None)
    if vp:
        st._visited_places = {tuple(np.round(np.asarray(k, dtype=float) - vec, 4).tolist())
                              for k in vp}
        moved += 1
    return {"moved": moved}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="l5.pkl")
    ap.add_argument("--by", choices=("x", "median"), default="x")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    st = Stream()
    if not persist.load_l5(st, name=a.name):
        print(f"没有存档：{a.name}")
        return 1

    before = fingerprint(st)
    vec = choose_vector(st, a.by)
    print(f"搬的向量：‖vec‖ = {np.linalg.norm(vec):,.3f}（按 {a.by}）")
    print(f"搬之前：|x| = {np.linalg.norm(st._st.x):,.3f}，"
          f"regions = {len(st.landscape.state_space.attractors())}")

    shift_in_place(st, vec)
    after = fingerprint(st)
    # 逐位相等做不到：从 1e6 量级的数上减一个 1e6 量级的数，float64 本身要丢 1e-10。
    # 所以用相对容差（1e-9），并把「丢了多少」报出来——这本身也是「坐标不该这么大」的证据。
    scale = float(np.linalg.norm(vec)) or 1.0
    tol = 1e-12 * scale
    worst = float(np.max(np.abs(before - after))) if before.size and after.size else 0.0
    same = before.shape == after.shape and worst <= tol
    print(f"搬之后：|x| = {np.linalg.norm(st._st.x):,.3f}，"
          f"visited_places = {len(getattr(st, '_visited_places', set()) or ())}")
    print(f"几何指纹：{len(before)} 对，最大差 {worst:.3e}（容差 {tol:.3e}）→ "
          f"{'一致' if same else '不一致'}")

    if not same:
        print("!! 几何变了，别用这份存档 —— 说明还有绝对坐标没搬全。")
        return 2

    if a.apply:
        out = f"l5.recentered-{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        persist.save_l5(st, name=out)
        print(f"已另存：data/state/{out}（没有覆盖 {a.name}）")
    else:
        print("（dry-run，没写任何东西；要存就加 --apply）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
