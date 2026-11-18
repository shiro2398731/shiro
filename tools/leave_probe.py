r"""tools/leave_probe.py — 副本实验：她「离开」时到底该挪多远，才真的换地方。

背景（HANDOFF 第 80 节）
------------------------
她的步长 `s` = 所有区域 sigma 的中位（副本 13:38 实测 618），而按所有区域算，
她到其它区域的距离中位是 91 万（同一个副本；上午真机是 127 万，那是均值/另一时刻）。
一步 618 只覆盖 0.05%。第 80 节的结论：**不是走一步，是跳过去**。

这一步量三种「离开」的后果（副本上跑，不碰真机）：

    baseline      src 原样：朝候选区域挪 min(s, dist)
    jump          直接放到**目标区域的质心**（瞬移）
    prop10        朝目标方向挪 dist * 0.10（一步跨出去，但不瞬移）
    prop10_nocap  prop10 + 关掉两层 4× 锚定（诊断：只有目标很远时才有差别）

目标 = `ActionGate.direction()` 现在选的那个候选区域（优先「没去过的、够得着的」；
没有就选最近的、至少一个半径远的）。本工具**不重新发明选目标**：它调用 src 的
`direction()`，再按 src 同一段逻辑复算选中的是哪个候选，并用返回向量校验
（`replication_mismatch` 计数；不为 0 就说明 src 改了、这个实验要跟着改）。

另一个**诊断**变体（不在三变体里，用来定位环的机制）：

    baseline_fixvisit  只修「去过」记账（按 uid 记，而不是按 4 位小数的位置），
                       走法仍是 baseline。用来回答：环是「走不到」还是「选目标」？

为什么 prop10 要配一个 nocap
----------------------------
src 有两层 4× 锚定（`_apply_leave` 末尾 + `Stream.step` 里）：
离最近质心超过 4*s ≈ 2472 就拉回最近质心。prop10 若一步挪 6-12 万，会被拉回**出发地**
（净位移 0）。`prop10_nocap` 关掉这两层，才量得到这个变体本来的后果。
在本副本上目标只有 617 远，10% = 62 < 2472，所以两者数值应当相同——这是对账点。

量什么
------
- 转移图：nodes（转移涉及的 uid 数）、edges（去重有向边）、
  back（A→B→A 的比例）、transitions（区域改变次数）；
- 不同区域数、`_visited_places` 大小、新区域数（`_next_uid` 增量）；
- 每一步的 |x|（CSV），每 50 步的地形散布（CSV）：区域数、mu 范数中位、
  mu 两两距离中位（随机抽 20,000 对；精确两两 2338² 太慢）；
- leave_rate、**所选目标距离**的分布（这是关键：一步够不够得着**目标**，
  不是「所有区域的中位距离」）、到达率（那一步结束时是否真在目标区域）、
  每次 leave 的位移中位。

硬约束
------
只读 `data/exp/base_1338.pkl`（她 13:38 的存档快照），只写 `data/exp/` 下的
CSV/JSON/SVG。**不写 data/state，不调用 save_l5，不启停任何进程。**

副本里 `rehearse_every = 0`：预演在沙盒副本里跑、不写回，不改变她这一步的轨迹
（exploration_gain 的生产用途是放大噪声，而这里的噪声是固定的外部输入），
关掉只省开销、少一个 deepcopy 整具身体的变量。

用法
----
    .\.venv\Scripts\python.exe tools\leave_probe.py
    .\.venv\Scripts\python.exe tools\leave_probe.py --variants baseline jump prop10
    .\.venv\Scripts\python.exe tools\leave_probe.py --variants baseline_fixvisit
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import textwrap
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from types import MethodType

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.L5_cognition.stream as stream_mod  # noqa: E402
from src.L5_cognition.action import LEAVE  # noqa: E402
from src.L5_cognition.dynamics import DynamicState  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from src.common import persist  # noqa: E402

DEFAULT_COPY = ROOT / "data/exp/base_1338.pkl"
DEFAULT_OUT = ROOT / "data/exp"
DEFAULT_STEPS = 600
DEFAULT_SEEDS = (0, 1)
DEFAULT_AMPLITUDE = 0.02          # 与 tools/rehearse.py / rollout_audit.py 的默认一致
TERRAIN_EVERY = 50
PAIR_SAMPLES = 20000              # 地形两两距离的抽样对数
PROP = 0.10                       # prop10 的比例
ANCHOR_FACTOR = 4.0               # src 的锚定倍数（只在文档里对账用）

MAIN_VARIANTS = ("baseline", "jump", "prop10", "prop10_nocap")
DIAG_VARIANTS = ("baseline_fixvisit",)
VARIANTS = MAIN_VARIANTS + DIAG_VARIANTS


# ---------------------------------------------------------------- 加载副本
def load_copy(path: Path) -> dict:
    """读副本 payload。拒绝 data/state 下的路径——真机存档连读都不该走这里。"""
    p = Path(path).resolve()
    state_dir = (ROOT / "data/state").resolve()
    if state_dir in p.parents:
        raise SystemExit(f"refusing to touch live state: {p}")
    with p.open("rb") as f:
        return pickle.load(f)


def make_stream(payload: dict, seed: int, amplitude: float) -> Stream:
    """一具副本身体：构造 -> apply payload -> 给上游噪声。

    payload 不恢复 RNG（存档里就没有，见 sandbox.py 文档），所以副本的噪声从
    seed 重新开始——这正是「seed 控制噪声」的地方。
    """
    dim = int(np.asarray(payload["x"]).ravel().size)
    st = Stream(dim=dim, seed=int(seed))
    persist.apply_l5_payload(st, payload)
    st.landscape.set_noise_amplitude(float(amplitude))
    # 副本实验关掉「预演」。理由：预演在沙盒副本里跑、不写回，不改变她这一步的
    # 轨迹（exploration_gain 的生产用途是噪声放大，而这里噪声是固定的外部输入）；
    # 关掉只省开销、少一个会 deepcopy 整具身体的变量。她自己的四步循环条件不受影响。
    st.rehearse_every = 0
    return st


# ------------------------------------------------- 关掉 step() 里的 4× 锚定
_NOCAP_STEP = None


def _no_anchor_step():
    """`Stream.step` 的源码副本，只把 4× 锚定的条件改成 False。

    不改 src 文件：`inspect.getsource` 拿源码 -> 文本级替换一处 ->
    `exec` 到 stream 模块命名空间 -> 临时赋给类。只有 prop10_nocap 用。
    """
    global _NOCAP_STEP
    if _NOCAP_STEP is not None:
        return _NOCAP_STEP
    src = textwrap.dedent(__import__("inspect").getsource(Stream.step))
    needle = "if _scale > 0 and float(_d[_j]) > 4.0 * _scale:"
    if src.count(needle) != 1:
        raise RuntimeError(f"step() 源码变了：找不到唯一的锚定条件（{src.count(needle)}）")
    patched = src.replace(
        needle, "if False:  # 4x anchor disabled by leave_probe (prop10_nocap)")
    ns = dict(stream_mod.__dict__)
    exec(compile(patched, "<leave_probe:step>", "exec"), ns)  # noqa: S102
    _NOCAP_STEP = ns["step"]
    return _NOCAP_STEP


@contextmanager
def step_without_anchor():
    orig = Stream.step
    Stream.step = _no_anchor_step()
    try:
        yield
    finally:
        Stream.step = orig


# ------------------------------------------------- 复刻 direction 的选目标逻辑
def _selection(x: np.ndarray, scale: float, others: np.ndarray,
               places, visited) -> dict | None:
    """按 src `ActionGate.direction()` 的分支逻辑复算「选了哪个候选」。

    返回 {branch, j, dist, step}。j 是 others 里的下标。
    **这是复制，不是发明**：与 src 的差别只在返回 j 而不是向量；调用方必须用
    `_verify` 拿 direction 实际返回的向量对账。
    """
    s = float(scale)
    x = np.asarray(x, dtype=float)
    cand = np.asarray(others, dtype=float)
    d = cand - x
    dist = np.linalg.norm(d, axis=1)
    if not np.isfinite(dist).any():
        return None
    ok = dist[np.isfinite(dist)]
    reach = float(np.median(ok)) if ok.size >= 2 else s

    keys = list(places) if places is not None else []
    if keys:
        vis = set(visited) if visited is not None else set()
        fresh = [k for k in range(len(keys))
                 if keys[k] not in vis and np.isfinite(dist[k]) and 0 < dist[k] <= reach]
        if fresh:
            j = int(min(fresh, key=lambda q: dist[q]))
            return {"branch": "fresh", "j": j, "dist": float(dist[j]),
                    "step": min(s, float(dist[j]))}

    pos = np.where(np.isfinite(dist) & (dist >= s))[0]
    if pos.size:
        j = int(pos[np.argmin(dist[pos])])
        return {"branch": "fallback", "j": j, "dist": float(dist[j]),
                "step": min(s, float(dist[j]))}
    return None


def _verify(d, x: np.ndarray, others: np.ndarray, sel: dict | None) -> bool:
    """direction 实际返回的向量，是否与复刻选中的那个候选一致。"""
    if sel is None or d is None:
        return False
    v = np.asarray(others, dtype=float)[sel["j"]] - np.asarray(x, dtype=float)
    nv = float(np.linalg.norm(v))
    dd = np.asarray(d, dtype=float)
    nd = float(np.linalg.norm(dd))
    if nv <= 0 or nd <= 0:
        return False
    if abs(nd - sel["step"]) > 1e-9 * max(1.0, sel["step"]):
        return False
    return float((v / nv) @ (dd / nd)) > 1.0 - 1e-9


def _uid_for_mu(st: Stream, mu: np.ndarray) -> int:
    attrs = st.landscape.state_space.attractors()
    mus = np.stack([np.asarray(a.mu, dtype=float) for a in attrs])
    target = np.asarray(mu, dtype=float)
    equals = np.where(np.all(mus == target, axis=1))[0]
    row = int(equals[0]) if equals.size else int(np.argmin(np.linalg.norm(
        mus - target, axis=1)))
    return int(attrs[row].uid)


def _key_in_visited(mu, places, visited):
    """候选区域的 4 位小数位置键，是否已经在 `_visited_places` 里。"""
    if places is None or visited is None:
        return None
    key = tuple(np.round(np.asarray(mu, dtype=float), 4).tolist())
    return bool(key in set(visited))


def _nearest_visited_dist(mu, visited):
    """候选质心到最近一个「踩过的位置」的距离——比 4 位小数的量子（1e-4）大还是小。"""
    if not visited:
        return None
    arr = np.asarray(list(visited), dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    return float(np.linalg.norm(arr - np.asarray(mu, dtype=float), axis=1).min())


def record_direction(st: Stream) -> None:
    """给 `st.actions.direction` 套记录器：记下这一步选中的目标（uid/距离/分支）。

    记录器调用原函数；再用 `_selection` 复算、用返回向量校验。校验失败才退回
    按方向找最近候选（并计数，写进 summary 的 replication_mismatch）。
    """
    orig = st.actions.direction
    st._probe_mismatch = 0

    def recorder(x, centroid, scale, neighbours=None, visited=None, places=None):
        d = orig(x, centroid, scale, neighbours=neighbours, visited=visited,
                 places=places)
        st._probe_last = None
        if d is None or neighbours is None:
            return d
        nb = np.asarray(neighbours, dtype=float)
        if nb.ndim != 2 or len(nb) == 0:
            return d
        x = np.asarray(x, dtype=float)
        sel = _selection(x, scale, nb, places, visited)
        if _verify(d, x, nb, sel):
            j = sel["j"]
        else:
            # 退路：按返回向量的方向找候选（可能与 src 的并列打破规则不同）
            st._probe_mismatch += 1
            dd = nb - x
            dist = np.linalg.norm(dd, axis=1)
            unit = np.asarray(d, dtype=float) / float(np.linalg.norm(d))
            safe = np.where(dist > 0, dist, np.inf)
            dots = (dd / safe[:, None]) @ unit
            dots[~np.isfinite(dots)] = -np.inf
            j = int(np.argmax(dots))
            sel = {"branch": "align", "j": j,
                   "dist": float(dist[j]), "step": float(np.linalg.norm(d))}
        st._probe_last = {
            "mu": nb[j].copy(),
            "uid": _uid_for_mu(st, nb[j]),
            "dist": float(sel["dist"]),
            "step": float(sel["step"]),
            "branch": sel["branch"],
            "in_visited": _key_in_visited(nb[j], places, visited),
            "nearest_visited_key_dist": _nearest_visited_dist(nb[j], visited),
        }
        return d

    st.actions.direction = recorder


# ---------------------------------------------------------- 变体的 _apply_leave
def _anchor(self, space, scale) -> None:
    """src `_apply_leave` 末尾的 4× 锚定，逐行照抄（只有注释不同）。"""
    try:
        _attrs = space.attractors()
        if _attrs:
            _mus = np.stack([np.asarray(a.mu, dtype=float) for a in _attrs])
            _d = np.linalg.norm(_mus - np.asarray(self._st.x, dtype=float), axis=1)
            _j = int(np.argmin(_d))
            _lim = ANCHOR_FACTOR * float(scale) if scale else 0.0
            if _lim > 0 and float(_d[_j]) > _lim:
                self._st = DynamicState(x=_mus[_j].copy(), v=self._st.v)
    except Exception:  # noqa: BLE001
        pass


def _variant_leave(self, mode: str, anchors: bool) -> None:
    """jump / prop10 / baseline_fixvisit：只换「去哪、挪多远」。"""
    node = self.detector.current_attractor
    space = self.landscape.state_space
    row = None if node is None else space.uid_row().get(int(node))
    if row is None:
        return
    scale = self._own_scale()
    if scale is None:
        return
    attrs = space.attractors()
    others = None
    places = None
    idx = None
    if len(attrs) > 1:
        idx = [i for i in range(len(attrs)) if i != row]
        others = np.stack([np.asarray(attrs[i].mu, dtype=float) for i in idx])
        places = [tuple(np.round(np.asarray(attrs[i].mu, dtype=float), 4).tolist())
                  for i in idx]

    visited = self._visited_places
    if mode == "baseline_fixvisit":
        # 只修「去过」记账：把已经踩过的区域按 **uid** 标成 visited（用当前质心的键），
        # 其余一切与 baseline 相同。这是诊断「环是不是选目标造成的」。
        seen_uids = getattr(self, "_probe_seen_uids", set())
        by_uid = {int(attrs[i].uid): places[k] for k, i in enumerate(idx)}
        visited = set(self._visited_places) | {by_uid[u] for u in seen_uids
                                               if u in by_uid}

    d = self.actions.direction(self._st.x, attrs[row].mu, scale,
                               neighbours=others, visited=visited, places=places)
    if d is None:
        return
    x = np.asarray(self._st.x, dtype=float)
    nx = x + d                          # 兜底：按 baseline 走
    tgt_mu = None
    tgt_uid = None
    dec_dist = None
    if others is not None:
        # 本地复算 direction 选中的候选，再用实际返回向量校验（不依赖记录器状态）。
        sel = _selection(x, scale, others, places, visited)
        if _verify(d, x, others, sel):
            tgt_mu = np.asarray(others[sel["j"]], dtype=float)
            tgt_uid = int(attrs[idx[sel["j"]]].uid)
            dec_dist = float(sel["dist"])
        elif getattr(self, "_probe_last", None) is not None:
            # 退路：用记录器在这次调用里记下的目标（再校验它确实是某个候选）。
            p = self._probe_last
            eq = np.where(np.all(np.asarray(others) == np.asarray(p["mu"], float),
                                 axis=1))[0]
            if eq.size:
                tgt_mu = np.asarray(others[int(eq[0])], dtype=float)
                tgt_uid = int(p["uid"])
                dec_dist = float(p.get("dist", np.linalg.norm(tgt_mu - x)))

    if mode == "jump" and tgt_mu is not None:
        nx = tgt_mu.copy()
    elif mode == "prop10" and tgt_mu is not None:
        dist = float(np.linalg.norm(tgt_mu - x))
        nd = float(np.linalg.norm(d))
        if nd > 0:
            nx = x + d / nd * (dist * PROP)
    elif mode == "baseline_fixvisit" and tgt_uid is not None:
        seen = getattr(self, "_probe_seen_uids", set())
        seen.add(int(tgt_uid))
        self._probe_seen_uids = seen

    if tgt_mu is not None and tgt_uid is not None:
        if dec_dist is None:
            dec_dist = float(np.linalg.norm(tgt_mu - x))
        prev = getattr(self, "_probe_last", None) or {}
        self._probe_last = {
            "mu": np.asarray(tgt_mu, dtype=float).copy(),
            "uid": int(tgt_uid),
            "dist": float(dec_dist),
            "branch": str(prev.get("branch", mode)),
            "in_visited": _key_in_visited(tgt_mu, places, visited),
            "nearest_visited_key_dist": _nearest_visited_dist(
                tgt_mu, getattr(self, "_visited_places", None)),
        }

    self._st = DynamicState(x=nx, v=self._st.v)
    try:
        _here = tuple(np.round(np.asarray(self._st.x, dtype=float), 4).tolist())
        self._visited_places.add(_here)
    except Exception:  # noqa: BLE001
        pass
    if anchors:
        _anchor(self, space, scale)


def install_variant(st: Stream, mode: str, anchors: bool) -> None:
    st._apply_leave = MethodType(
        partial(_variant_leave, mode=mode, anchors=anchors), st)


# ---------------------------------------------------------------- 地形采样
def terrain_sample(st: Stream, rng: np.random.Generator, step: int) -> dict:
    attrs = st.landscape.state_space.attractors()
    n = len(attrs)
    if n == 0:
        return {"step": int(step), "n_regions": 0,
                "mu_norm_median": float("nan"),
                "mu_pair_median": float("nan"), "n_pairs": 0}
    mus = np.stack([np.asarray(a.mu, dtype=float) for a in attrs])
    norms = np.linalg.norm(mus, axis=1)
    k = min(PAIR_SAMPLES, max(1, n * (n - 1)))
    i = rng.integers(0, n, size=k)
    j = rng.integers(0, n, size=k)
    keep = i != j
    i, j = i[keep], j[keep]
    pair_d = np.linalg.norm(mus[i] - mus[j], axis=1)
    return {"step": int(step), "n_regions": int(n),
            "mu_norm_median": float(np.median(norms)),
            "mu_pair_median": float(np.median(pair_d)) if len(pair_d) else float("nan"),
            "n_pairs": int(len(pair_d))}


# ------------------------------------------------- 起始几何：一步够不够得着目标
def geometry(st: Stream) -> dict:
    """她此刻的候选几何：所有区域距离的分布、最近候选、一步能到哪、visited 覆盖。"""
    space = st.landscape.state_space
    attrs = space.attractors()
    x = np.asarray(st._st.x, dtype=float)
    mus = np.stack([np.asarray(a.mu, dtype=float) for a in attrs])
    d = np.linalg.norm(mus - x, axis=1)
    s = st._own_scale()
    own = int(np.argmin(d))
    other_idx = [i for i in range(len(attrs)) if i != own]
    d_other = d[other_idx]
    keys = [tuple(np.round(mus[i], 4).tolist()) for i in other_idx]
    vis = set(st._visited_places)
    reach = float(np.median(d_other)) if d_other.size else float("nan")
    fresh = [k for k in range(len(keys))
             if keys[k] not in vis and np.isfinite(d_other[k])
             and 0 < d_other[k] <= reach]
    out = {
        "n_regions": int(len(attrs)),
        "s_median_sigma": round(float(s), 4) if s else None,
        "abs_x": round(float(np.linalg.norm(x)), 4),
        "cand_dist_p10": round(float(np.percentile(d_other, 10)), 3),
        "cand_dist_p50": round(float(np.percentile(d_other, 50)), 3),
        "cand_dist_p90": round(float(np.percentile(d_other, 90)), 3),
        "cand_dist_mean": round(float(np.mean(d_other)), 3),
        "nearest_other": round(float(d_other.min()), 3) if d_other.size else None,
        "nearest_ge_s": None,
        "reach_median": round(float(reach), 3) if np.isfinite(reach) else None,
        "n_fresh_within_reach": int(len(fresh)),
        "nearest_fresh": round(float(min((d_other[k] for k in fresh),
                                         default=float("nan"))), 3),
        "centroids_in_visited": int(sum(1 for k in keys if k in vis)),
        "visited_places": int(len(vis)),
    }
    pos = np.where(d >= float(s))[0] if s else np.array([], dtype=int)
    if pos.size:
        out["nearest_ge_s"] = round(float(d[pos].min()), 3)
    return out


# ---------------------------------------------------------------- 转移图
def graph_metrics(seq: list[int]) -> dict:
    """nodes/edges/back/transitions。back = 下一步立刻回到上一步之前那个区域的比例。"""
    nodes: set[int] = set()
    edges: set[tuple[int, int]] = set()
    trans = 0
    back = 0
    for t in range(1, len(seq)):
        a, b = seq[t - 1], seq[t]
        if a < 0 or b < 0 or a == b:
            continue
        trans += 1
        nodes.update((a, b))
        edges.add((a, b))
        if t >= 2 and seq[t - 2] == b:
            back += 1
    return {
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "transitions": int(trans),
        "back_rate": round(back / trans, 4) if trans else 0.0,
        "distinct_regions": len({s for s in seq if s >= 0}),
    }


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(values, q)), 6)


# ---------------------------------------------------------------- 跑一个 case
def run_case(payload_path: Path, variant: str, seed: int, steps: int,
             amplitude: float, outdir: Path, quiet: bool = False) -> dict:
    payload = load_copy(payload_path)           # 每个 case 重新读，副本互不污染
    st = make_stream(payload, seed, amplitude)
    space = st.landscape.state_space
    n0 = len(space.attractors())
    uid0 = int(getattr(space, "_next_uid", 0))
    x0 = float(np.linalg.norm(np.asarray(st._st.x, dtype=float)))
    visited0 = len(st._visited_places)
    actions0 = {int(k): int(v) for k, v in (st.action_counts or {}).items()}
    geo0 = geometry(st)

    record_direction(st)
    anchors = variant != "prop10_nocap"
    if variant in ("jump", "prop10", "prop10_nocap", "baseline_fixvisit"):
        # prop10_nocap 的移动方式和 prop10 一样，只是关掉两层锚定。
        mode = "prop10" if variant == "prop10_nocap" else variant
        install_variant(st, mode=mode, anchors=anchors)

    terrain_rng = np.random.default_rng(20261118 + 7 * int(seed))
    terrain = [terrain_sample(st, terrain_rng, 0)]

    trace: list[tuple[int, float, int, int]] = []      # step, |x|, region, action
    leaves: list[dict] = []
    ctx = step_without_anchor() if variant == "prop10_nocap" else nullcontext()
    with ctx:
        for t in range(1, int(steps) + 1):
            st._probe_last = None
            x_before = np.asarray(st._st.x, dtype=float).copy()
            st.step()
            x = np.asarray(st._st.x, dtype=float)
            aid = st.detector.current_attractor
            aid = -1 if aid is None else int(aid)
            act = int(getattr(st, "_action", -1))
            disp = float(np.linalg.norm(x - x_before))
            trace.append((t, float(np.linalg.norm(x)), aid, act))
            if act == LEAVE:
                p = getattr(st, "_probe_last", None)
                if p is not None:
                    leaves.append({
                        "step": int(t),
                        "target_uid": int(p["uid"]),
                        "target_dist": float(p["dist"]),
                        "branch": str(p["branch"]),
                        "arrived": aid == int(p["uid"]),
                        "disp": disp,
                        "in_visited": bool(p.get("in_visited") or False),
                        "nearest_visited_key_dist": p.get("nearest_visited_key_dist"),
                        "mu": np.asarray(p["mu"], dtype=float).copy(),
                    })
            if t % TERRAIN_EVERY == 0:
                terrain.append(terrain_sample(st, terrain_rng, t))

    seq = [r[2] for r in trace]
    gm = graph_metrics(seq)
    n_end = len(space.attractors())
    uid_end = int(getattr(space, "_next_uid", 0))
    n_leaves = len(leaves)
    actions = st.action_counts or {}
    hold_n = int(actions.get(0, 0)) - int(actions0.get(0, 0))
    leave_n = int(actions.get(1, 0)) - int(actions0.get(1, 0))
    total_actions = hold_n + leave_n
    tdist = [x["target_dist"] for x in leaves]
    s = st._own_scale()
    # 「同一个区域再次被选中时，它的质心挪了多远」——4 位小数键的量子是 1e-4，
    # 质心挪动超过这个数，旧位置键就对不上了（环的机制，见文档）。
    same_drift: list[float] = []
    prev_mu: dict[int, np.ndarray] = {}
    for x in leaves:
        u = int(x["target_uid"])
        if u in prev_mu:
            same_drift.append(float(np.linalg.norm(x["mu"] - prev_mu[u])))
        prev_mu[u] = x["mu"]
    nvk = [x["nearest_visited_key_dist"] for x in leaves
           if x["nearest_visited_key_dist"] is not None]
    case = {
        "variant": variant,
        "seed": int(seed),
        "steps": int(steps),
        "amplitude": float(amplitude),
        "anchors": bool(anchors),
        "start": {"n_regions": int(n0), "next_uid": uid0, "abs_x": round(x0, 4),
                  "visited_places": int(visited0)},
        "end": {"n_regions": int(n_end), "next_uid": uid_end,
                "abs_x": round(float(trace[-1][1]), 4) if trace else None,
                "abs_x_max": round(float(max(r[1] for r in trace)), 4) if trace else None,
                "visited_places": int(len(st._visited_places))},
        "new_regions": int(uid_end - uid0),
        "actions": {"hold": hold_n, "leave": leave_n,
                    "leave_rate": round(leave_n / total_actions, 4)
                    if total_actions else 0.0},
        "replication_mismatch": int(getattr(st, "_probe_mismatch", 0)),
        "leaves_measured": int(n_leaves),
        "target_dist_p10": _pct(tdist, 10),
        "target_dist_p50": _pct(tdist, 50),
        "target_dist_p90": _pct(tdist, 90),
        "target_within_s_rate": round(
            sum(1 for v in tdist if s and v <= float(s)) / n_leaves, 4)
        if n_leaves and s else 0.0,
        "target_uids": len({x["target_uid"] for x in leaves}),
        "target_key_in_visited_rate": round(
            sum(1 for x in leaves if x["in_visited"]) / n_leaves, 4)
        if n_leaves else 0.0,
        "nearest_visited_key_dist_p50": _pct(nvk, 50),
        "same_target_mu_drift_p50": _pct(same_drift, 50),
        "same_target_mu_drift_max": round(float(max(same_drift, default=0.0)), 6),
        "arrival_rate": round(sum(1 for x in leaves if x["arrived"]) / n_leaves, 4)
        if n_leaves else 0.0,
        "leave_disp_median": _pct([x["disp"] for x in leaves], 50),
        "leave_disp_max": round(float(max((x["disp"] for x in leaves), default=0.0)), 4),
        "terrain_start": terrain[0],
        "terrain_end": terrain[-1],
        "geometry_start": geo0,
        "geometry_end": geometry(st),
        **gm,
    }

    run = f"{variant}_s{int(seed)}"
    tr_path = outdir / f"leave_probe_{run}_trace.csv"
    with tr_path.open("w", encoding="utf-8", newline="") as f:
        f.write("step,abs_x,region,action\n")
        for t, x_norm, aid, act in trace:
            f.write(f"{t},{x_norm:.6f},{aid},{act}\n")
    te_path = outdir / f"leave_probe_{run}_terrain.csv"
    with te_path.open("w", encoding="utf-8", newline="") as f:
        f.write("step,n_regions,mu_norm_median,mu_pair_median,n_pairs\n")
        for row in terrain:
            f.write(f"{row['step']},{row['n_regions']},"
                    f"{row['mu_norm_median']:.6f},{row['mu_pair_median']:.6f},"
                    f"{row['n_pairs']}\n")
    lv_path = outdir / f"leave_probe_{run}_leaves.csv"
    with lv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("step,target_uid,target_dist,branch,arrived,disp,"
                "in_visited,nearest_visited_key_dist\n")
        for row in leaves:
            nv = row["nearest_visited_key_dist"]
            f.write(f"{row['step']},{row['target_uid']},{row['target_dist']:.6f},"
                    f"{row['branch']},{int(row['arrived'])},{row['disp']:.6f},"
                    f"{int(row['in_visited'])},"
                    f"{'nan' if nv is None else f'{nv:.8f}'}\n")
    case["trace_csv"] = str(tr_path.relative_to(ROOT))
    case["terrain_csv"] = str(te_path.relative_to(ROOT))
    case["leaves_csv"] = str(lv_path.relative_to(ROOT))
    case["terrain_curve"] = terrain
    if not quiet:
        print(f"[{run}] nodes={case['graph_nodes']} edges={case['graph_edges']} "
              f"back={case['back_rate']} trans={case['transitions']} "
              f"distinct={case['distinct_regions']} |x|={case['end']['abs_x']:.0f} "
              f"leave={leave_n} tgt_p50={case['target_dist_p50']} "
              f"arrival={case['arrival_rate']} new_regions={case['new_regions']} "
              f"mismatch={case['replication_mismatch']}", flush=True)
    return case


# ---------------------------------------------------------------- 曲线 SVG
def _read_trace(path: Path) -> list[tuple[int, float, int, int]]:
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        next(f, None)
        for line in f:
            a = line.strip().split(",")
            if len(a) == 4:
                out.append((int(a[0]), float(a[1]), int(a[2]), int(a[3])))
    return out


def _polyline(points: list[tuple[float, float]], color: str, dash: str = "") -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline fill="none" stroke="{color}" stroke-width="1.6"{d} '
            f'points="{pts}"/>')


def write_svg(cases: list[dict], path: Path) -> None:
    """两栏曲线：左 |x|(step)，右地形 mu 两两距离中位(step)。手写 SVG，不装库。"""
    colors = {"baseline": "#7f7f7f", "jump": "#1f77b4", "prop10": "#d62728",
              "prop10_nocap": "#ff7f0e", "baseline_fixvisit": "#2ca02c"}
    W, H, PAD = 980, 360, 52
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" font-family="monospace" font-size="11">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    for pi, key in enumerate(("abs_x", "spread")):
        x0 = 20 + pi * (W // 2)
        w, h = W // 2 - 46, H - 2 * PAD
        series = []
        for c in cases:
            if key == "abs_x":
                rows = _read_trace(ROOT / c["trace_csv"])
                pts = [(r[0], r[1]) for r in rows]
            else:
                pts = [(r["step"], r["mu_pair_median"]) for r in c["terrain_curve"]]
            series.append((c, pts))
        xs = [p[0] for _, pts in series for p in pts]
        ys = [p[1] for _, pts in series for p in pts]
        if not xs:
            continue
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0
        sx = lambda v: x0 + (v - xmin) / (xmax - xmin) * w       # noqa: E731
        sy = lambda v: PAD + h - (v - ymin) / (ymax - ymin) * h  # noqa: E731
        title = "|x| vs step" if key == "abs_x" else "median |mu_i - mu_j| vs step"
        parts.append(f'<text x="{x0}" y="26" fill="#222">{title}</text>')
        parts.append(f'<line x1="{x0}" y1="{PAD + h}" x2="{x0 + w}" '
                     f'y2="{PAD + h}" stroke="#ccc"/>')
        parts.append(f'<text x="{x0 - 6}" y="{PAD + 8}" text-anchor="end" '
                     f'fill="#666">{ymax:.3g}</text>')
        parts.append(f'<text x="{x0 - 6}" y="{PAD + h}" text-anchor="end" '
                     f'fill="#666">{ymin:.3g}</text>')
        for c, pts in series:
            if not pts:
                continue
            color = colors.get(c["variant"], "#000000")
            dash = "5,4" if c["seed"] == 1 else ""
            parts.append(_polyline([(sx(p[0]), sy(p[1])) for p in pts], color, dash))
        used = [v for v in VARIANTS if any(c["variant"] == v for c in cases)]
        for li, v in enumerate(used):
            parts.append(f'<rect x="{x0 + li * 120}" y="{PAD + h + 8}" width="18" '
                         f'height="3" fill="{colors.get(v, "#000")}"/>')
            parts.append(f'<text x="{x0 + li * 120 + 22}" y="{PAD + h + 14}" '
                         f'fill="#444">{v}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------- 打印总结
def print_tables(cases: list[dict]) -> None:
    print("\n## 转移图与位置（一行 = 一个变体 × 一个 seed）\n")
    print("| variant | seed | nodes | edges | back | trans | distinct | |x|_end | "
          "|x|_max | new_regions | leave | leave_rate | tgt_p50 | within_s | arrival |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cases:
        print(f"| {c['variant']} | {c['seed']} | {c['graph_nodes']} | "
              f"{c['graph_edges']} | {c['back_rate']} | {c['transitions']} | "
              f"{c['distinct_regions']} | {c['end']['abs_x']:.0f} | "
              f"{c['end']['abs_x_max']:.0f} | {c['new_regions']} | "
              f"{c['actions']['leave']} | {c['actions']['leave_rate']} | "
              f"{c['target_dist_p50']} | {c['target_within_s_rate']} | "
              f"{c['arrival_rate']} |")
    print("\n## 环的机制（visited 记账是否有效；4 位小数键的量子 = 1e-4）\n")
    print("| variant | seed | tgt_key_in_visited | nearest_visited_dist_p50 | "
          "same_target_mu_drift_p50 | target_uids |")
    print("|---|---|---|---|---|---|")
    for c in cases:
        print(f"| {c['variant']} | {c['seed']} | {c['target_key_in_visited_rate']} | "
              f"{c['nearest_visited_key_dist_p50']} | "
              f"{c['same_target_mu_drift_p50']} | {c['target_uids']} |")
    print("\n## 地形散布（每 50 步采样；seed 0）\n")
    for v in VARIANTS:
        cs = [c for c in cases if c["variant"] == v and c["seed"] == 0]
        if not cs:
            continue
        print(f"### {v}")
        print("| step | n_regions | mu_norm_med | mu_pair_med |")
        print("|---|---|---|---|")
        for row in cs[0]["terrain_curve"]:
            print(f"| {row['step']} | {row['n_regions']} | "
                  f"{row['mu_norm_median']:.0f} | {row['mu_pair_median']:.0f} |")
        print()


# ---------------------------------------------------------------- main
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="副本实验：离开的三种挪法")
    ap.add_argument("--copy", type=Path, default=DEFAULT_COPY,
                    help="副本 payload（默认 data/exp/base_1338.pkl）")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    ap.add_argument("--variants", nargs="+", default=list(MAIN_VARIANTS),
                    choices=list(VARIANTS))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    if not args.copy.exists():
        raise SystemExit(f"copy not found: {args.copy}")
    args.out = args.out.resolve()
    args.copy = args.copy.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    cases = []
    for variant in args.variants:
        for seed in args.seeds:
            cases.append(run_case(args.copy, variant, seed, args.steps,
                                  args.amplitude, args.out))

    summary = {
        "copy": str(args.copy),
        "steps": args.steps,
        "seeds": list(args.seeds),
        "amplitude": args.amplitude,
        "terrain_every": TERRAIN_EVERY,
        "pair_samples": PAIR_SAMPLES,
        "prop": PROP,
        "anchor_factor": ANCHOR_FACTOR,
        "cases": cases,
    }
    out_json = args.out / "leave_probe_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    svg_path = args.out / "leave_probe_curves.svg"
    write_svg(cases, svg_path)
    print_tables(cases)
    print(f"\nsummary: {out_json.relative_to(ROOT)}")
    print(f"curves : {svg_path.relative_to(ROOT)}")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
