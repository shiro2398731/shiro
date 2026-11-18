r"""坐标系归一：把「她」和「她的地形」一起搬回原点。

为什么会有这个东西
------------------
2026-11-18 下午，白默定的方向是：**「我不觉得应该有边界。」**
不设界，就只剩一条路——让「坐标有多大」永远不影响她能不能活。

她的空间里，动力学只用距离（`direction`、`gradient`、`tension` 全是距离/权重），
所以把 `x`、所有 `mu`、所有 `members`、`landscape._recent`、`_visited_places` 里的
位置键一起减掉同一个向量，**行为逐位不变**。变的只有浮点数的量级：

- `SAVE_MAGNITUDE_LIMIT = 1e6` 是分量绝对值上限，撞上就拒存（2026-11-18 08:24 崩了 4 次）；
- float64 从 1e6 量级上相减，本身要丢 1e-10（几何指纹能看到）。

什么时候搬
----------
**只在 boot 时**（journal replay 之后、开始推演之前）检查一次：
量级超过 `ORIGIN_LIMIT` 就搬回原点。

这不是边界。她不会因为「走太远」被拦住；下一次启动只是换个坐标系继续。
搬的是「我站在哪儿」这个说法，不是「我能走多远」这条规则。

2026-11-18 08:23 的 `recentered` 存档就是漏了一半：只搬了 `mu`、没搬 `x`，
她于是从「在云里」变成「在云外 194 万」。所以这里宁可搬得笨一点，也要搬全。
"""

from __future__ import annotations

import numpy as np

from .dynamics import DynamicState

# C1 工程参数（可以人为，但要写来源）：存档上限是 1e6，取它一个数量级以下，
# 留出至少 5 倍的余量给「两次 boot 之间她可能走的最远距离」。
# 它不决定她去哪，只决定「下次启动要不要换个原点」。不是 C3 主观内容。
ORIGIN_LIMIT = 200_000.0


def _arrays(stream) -> list:
    """所有「存着绝对坐标」的地方。少搬一个，她就会在搬完之后变成一个陌生人。"""
    sp = stream.landscape.state_space
    out = [np.asarray(stream._st.x, dtype=float)]
    for a in sp.attractors():
        out.append(np.asarray(a.mu, dtype=float))
    for r in list(getattr(stream.landscape, "_recent", ()) or ()):
        out.append(np.asarray(r, dtype=float))
    return out


def max_component(stream) -> float:
    """所有绝对坐标里最大的那一个分量（存档判据看的就是这个数）。"""
    best = 0.0
    for arr in _arrays(stream):
        if arr.size:
            best = max(best, float(np.abs(arr).max()))
    return best


def shift_origin(stream, vec) -> dict:
    """x / mu / members / _recent / _visited_places 一起减掉 vec。"""
    vec = np.asarray(vec, dtype=float)
    sp = stream.landscape.state_space
    moved = 0

    stream._st = DynamicState(x=np.asarray(stream._st.x, dtype=float) - vec,
                              v=np.asarray(stream._st.v, dtype=float))
    moved += 1

    for a in sp.attractors():
        a.mu = np.asarray(a.mu, dtype=float) - vec
        moved += 1
        members = getattr(a, "members", None)
        if members:
            a.members = type(members)(
                (np.asarray(m, dtype=float) - vec for m in members),
                maxlen=getattr(members, "maxlen", 64),
            )
            moved += 1

    recent = getattr(stream.landscape, "_recent", None)
    if recent:
        stream.landscape._recent = type(recent)(
            (np.asarray(r, dtype=float) - vec for r in list(recent)),
            maxlen=getattr(recent, "maxlen", 64),
        )
        moved += 1

    vp = getattr(stream, "_visited_places", None)
    if vp:
        stream._visited_places = {
            tuple(np.round(np.asarray(k, dtype=float) - vec, 4).tolist()) for k in vp}
        moved += 1

    return {"moved": moved, "vec_norm": float(np.linalg.norm(vec))}


def center_vector(stream) -> np.ndarray:
    """搬哪个向量：**地形的中心**（逐维中位），不是她站的地方。

    按 x 搬会让「她」归零、地形整体留在原点另一侧；按 mu 的中位搬，是让
    「她的世界」回到原点，她站在世界里原来的位置。后者才是「换坐标系」，
    前者只是「把她钉在原点上」。
    """
    mus = [np.asarray(a.mu, dtype=float) for a in stream.landscape.state_space.attractors()]
    if not mus:
        return np.asarray(stream._st.x, dtype=float).copy()
    return np.median(np.stack(mus), axis=0)


def maybe_reorigin(stream, limit: float = ORIGIN_LIMIT) -> dict:
    """量级太大就整体搬回原点；否则什么都不做。返回一份可以写进日志的账。"""
    before = max_component(stream)
    if before <= limit:
        return {"shifted": False, "max_component": before, "limit": limit}
    info = shift_origin(stream, center_vector(stream))
    after = max_component(stream)
    return {
        "shifted": True,
        "before": before,
        "after": after,
        "limit": limit,
        "vec_norm": info["vec_norm"],
        "moved": info["moved"],
        # 搬完还超限 = 她的地形本身就比上限大。那不是这次归一没做好，
        # 是「存档用绝对上限」这条设计到了该换的时候（见 docs/2026-11-18_存在_实账.md 第 2 刀）。
        "still_over": after > limit,
    }
