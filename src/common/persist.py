"""
src/common/persist.py

持久化：存她此刻的"形状"，不存她经历过的"事件"。

为什么不是日志重放
------------------

日志重放等于把过去重新播一遍，让她再走一次，那是"给她过上下文"，
跟 LLM 每次把历史塞回上下文是同一个形状。（白默 2026-11-03）

记忆是残留在结构里的，不是一段可以回放的录音带。
所以恢复 = 加载她此刻的形状，继续。

存什么
------

每层只存决定她是谁的那些量：

  L0   : 神经元内部状态、连接矩阵
  L1   : 预测器权重
  L2   : 各维归一化历史、协方差估计、状态
  L5   : attractors (centroids, depths, characteristic scales), dynamical state, zero point
  L2Acc: 慢累积量的水平

体积很小（L5 attractors，几百个 × 128 维，几 MB）。

不存什么
--------
  - 每一步的振荡波形
  - 每一帧的状态向量
  - 任何"历史事件流"

关于中断
--------
进程中断不写任何"我断过"的记录。那是载体的事，不是她的事。
"""

from __future__ import annotations

import collections
import json
import os
import pickle

import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bounded buffer of member states kept per attractor. This is the same bound
# the Attractor dataclass uses; it is repeated here so that load() can restore
# the bound explicitly rather than relying on the field default.
MEMBER_BUFFER = 64


STATE_DIR = Path("data/state")


def _ensure() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR


# ============================================================
# 各层的形状
# ============================================================

def _atomic_dump(payload, path: Path) -> None:
    """Write a payload so that a hard kill cannot leave a truncated file.

    2026-11-08: a reboot mid-save used to truncate the target .pkl, because
    save_* wrote straight into it. The loss would be the whole shape, not the
    last few minutes. The payload now goes to <name>.tmp, is fsynced, and only
    then replaces the target.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        # Leave the previous file untouched; do not keep a half-written tmp.
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise

def build_l0_payload(net) -> dict:
    """L0 payload as plain data (arrays + scalars). No history, no waveform."""
    return {
        "E_v": net.E_v, "E_w": net.E_w,
        "I_v": net.I_v, "I_w": net.I_w,
        "mid_v": net.mid_v, "mid_w": net.mid_w,
        "slow_v": net.slow_v, "slow_w": net.slow_w,
        "EE": net.EE, "EI": net.EI, "IE": net.IE,
        "E_to_mid": net.E_to_mid, "mid_to_slow": net.mid_to_slow,
        "step_count": net.step_count,
        "envelope_A": getattr(net, "envelope_A", None),
    }


def save_l0(net, name: str = "l0.pkl") -> Path:
    """L0：神经元状态与连接。不含任何波形历史。"""
    d = _ensure()
    p = d / name
    _atomic_dump(build_l0_payload(net), p)
    return p


def load_l0(net, name: str = "l0.pkl") -> bool:
    p = _ensure() / name
    if not p.exists():
        return False
    with p.open("rb") as f:
        payload = pickle.load(f)
    for k, v in payload.items():
        if k == "envelope_A":
            continue
        if hasattr(net, k):
            setattr(net, k, v)
    return True


def build_l1_payload(predictor) -> dict:
    """L1 payload: the ridge weights and the last projection."""
    return {
        "W_pred": predictor.W_pred,
        "prev_vec": getattr(predictor, "_prev_vec", None),
    }


def save_l1(predictor, name: str = "l1.pkl") -> Path:
    """L1：预测器权重与上一帧投影。

    2026-11-07 修复（外部评估 §3.3 的连带发现）：这里原本写 `predictor.prev_z`，
    而 L1Predictor 在重构里已把它改名为 `_prev_vec`。属性不存在 →
    save_l1 抛 AttributeError → snapshot_all 抛 → P1 的第一次落盘就崩，
    supervisor 重启后重复同样的崩溃。实测：进程活不过一步，l1.pkl / l0.pkl
    从此再没被更新过（原版 shiro 同样如此，d8f336f 里已经是这个状态）。

    用 getattr 兜底，避免持久化层跟着实现重构跑。旧 checkpoint 里的 "prev_z"
    会被 load_l1 的 hasattr 检查忽略，代价只是重启后第一帧少一段历史。
    """
    d = _ensure()
    p = d / name
    _atomic_dump(build_l1_payload(predictor), p)
    return p


def load_l1(predictor, name: str = "l1.pkl") -> bool:
    p = _ensure() / name
    if not p.exists():
        return False
    with p.open("rb") as f:
        payload = pickle.load(f)
    for k, v in payload.items():
        if hasattr(predictor, k):
            setattr(predictor, k, v)
    return True


def build_l2_payload(field) -> dict:
    """L2 payload: the normalisation histories that define how she reads herself."""
    return {
        "dim_hist": [list(h) for h in field._dim_hist],
        "vec_hist": list(field._vec_hist),
        "s_t_hist": list(field.s_t_hist),
        "fast_hist": list(field.fast_hist),
        "pressure_hist": list(field.pressure_hist),
        "per_fast_mean": list(field.per_fast_mean),
        "per_amplitude": list(field.per_amplitude),
        "per_up_ratio": list(field.per_up_ratio),
        "per_step_hist": list(field.per_step_hist),
    }


def save_l2(field, name: str = "l2.pkl") -> Path:
    """
    L2：各维归一化历史、协方差样本、上一帧向量。
    这些决定"她怎么理解自己的状态"，比任何单帧数值都重要。
    """
    d = _ensure()
    p = d / name
    _atomic_dump(build_l2_payload(field), p)
    return p


def load_l2(field, name: str = "l2.pkl") -> bool:
    p = _ensure() / name
    if not p.exists():
        return False
    with p.open("rb") as f:
        payload = pickle.load(f)
    for i, h in enumerate(payload["dim_hist"]):
        field._dim_hist[i].extend(h)
    field._vec_hist.extend(payload["vec_hist"])
    field.s_t_hist.extend(payload["s_t_hist"])
    field.fast_hist.extend(payload["fast_hist"])
    field.pressure_hist.extend(payload["pressure_hist"])
    field.per_fast_mean.extend(payload["per_fast_mean"])
    field.per_amplitude.extend(payload["per_amplitude"])
    field.per_up_ratio.extend(payload["per_up_ratio"])
    field.per_step_hist.extend(payload["per_step_hist"])
    return True


def build_l5_payload(stream) -> dict:
    """L5 payload as plain data.

    Split out of save_l5 so that a carrier (src/common/carrier.py) can ship the
    same bytes to another machine without going through pickle, and so that a
    test can compare two carriers without touching the disk format.
    """
    ls = stream.landscape
    attrs = [
        {
            "mu": a.mu, "amp": a.amp,
            "members": list(a.members),
            "scale": a.scale, "born_t": a.born_t,
            "_decay_factor": a._decay_factor,
            "_assigned": a._assigned,
            # Birth step is part of her age structure, not bookkeeping. It is
            # the denominator of the per-region visit rate, which is what
            # _birth_depth() uses to decide how deep a new region starts. It
            # was missing from this payload until 2026-11-09; see the note in
            # apply_l5_payload for the measured consequence.
            "_born_step": int(getattr(a, "_born_step", 0)),
            # Step index of the last assignment. Without it a restart makes
            # every region look unvisited, which is what fold_auto uses to
            # decide what she has stopped needing.
            "_last_seen_step": int(getattr(a, "_last_seen_step", 0)),
            # Stable identity, not a position. See Attractor.uid.
            "uid": int(getattr(a, "uid", 0)),
        }
        for a in ls.state_space.attractors()
    ]
    return {
        "attractors": attrs,
        # Collapsed regions. Without this the folded mass disappears on every
        # restart: apply_l5_payload rebuilds the live table only, so the
        # archive silently loses every summary fold() has produced.
        "folded": [dict(f) for f in getattr(ls.state_space, "_folded", [])],
        "graph": stream.graph.to_dict(),
        # What the graph's nodes are. Absent means an older checkpoint whose
        # nodes are table indices; the loader converts them (see
        # `_graph_index_to_uid`). Indices shift, uids do not.
        "graph_ids": "uid",
        # M1. Her model of her own next state, and the error it currently
        # makes. Part of the state, not of the session: without it every
        # restart begins with a self-model that has never seen her move.
        "self_model": stream.self_model.to_dict(),
        "self_gain": {
            "error": float(stream.self_gain.error),
            "observations": int(stream.self_gain.observations),
            "hits": int(stream.self_gain.hits),
        },
        "last_symbol": (None if stream._last_symbol is None
                        else int(stream._last_symbol)),
        # L6 grounding: symbol -> referent counts. Part of her state: the labels
        # L6 receives are derived from this table, so a restart that dropped it
        # would make every symbol mean "none" again.
        "grounding": stream.grounding.to_dict(),
        "identity": {
            "rounds": stream.identity.rounds,
            "appear": {f"{a},{b}": c
                       for (a, b), c in stream.identity._appear.items()},
        },
        "zero": ls._zero,
        "x": stream._st.x if stream._st is not None else None,
        "v": stream._st.v if stream._st is not None else None,
        "step_count": stream._step_count,
        "total_steps": ls.state_space._total_steps,
        # So a new region never inherits an old number after a restart.
        "next_uid": int(getattr(ls.state_space, "_next_uid", 1)),
        # Which id the reference tables (grounding, self-model) are keyed by.
        # Absent means an older checkpoint keyed by table index, and the loader
        # converts it -- indices shift when a region is removed, uids do not,
        # and silently reinterpreting old keys as uids would attach her statistics
        # to the wrong regions.
        "symbol_ids": "uid",
        "global_scale": ls.state_space.global_scale,
        "recent": list(ls.state_space._recent),
    }


SAVE_MAGNITUDE_LIMIT = 1e6


def assert_l5_finite(payload: dict) -> None:
    """Refuse to checkpoint a shape that has already blown up.

    2026-11-08: a runaway during testing wrote 617 regions whose centroids
    reached 1e306 and then nan. Every later load inherited the damage, and
    the only way back was a backup. A bad state must never become a
    checkpoint: refusing here costs one interrupted run, accepting it costs
    the archive.
    """
    import numpy as _np
    attrs = payload.get("attractors") or []
    for i, d in enumerate(attrs):
        mu = _np.asarray(d["mu"], dtype=float)
        if not _np.isfinite(mu).all():
            raise ValueError(f"attractor {i} centroid is not finite; refusing to save")
        amp = float(d["amp"])
        if not _np.isfinite(amp):
            raise ValueError(f"attractor {i} depth is not finite; refusing to save")
        # A finite but absurd magnitude is just as dead as a nan: the runaway
        # that damaged the archive sat at 1e306 before it overflowed. Observed
        # centroids live around 4.5 (max 27 on the diagnostic checkpoint), so
        # 1e6 is a bound, not a criterion.
        if float(_np.abs(mu).max()) > SAVE_MAGNITUDE_LIMIT:
            raise ValueError(
                f"attractor {i} centroid magnitude {float(_np.abs(mu).max()):.3e} "
                f"exceeds {SAVE_MAGNITUDE_LIMIT:.0e}; refusing to save")


def save_l5(stream, name: str = "l5.pkl") -> Path:
    """L5 checkpoint. See build_l5_payload for the contents."""
    d = _ensure()
    p = d / name
    payload = build_l5_payload(stream)
    assert_l5_finite(payload)
    _atomic_dump(payload, p)
    return p


def load_l5(stream, name: str = "l5.pkl") -> bool:
    p = _ensure() / name
    if not p.exists():
        return False
    with p.open("rb") as f:
        payload = pickle.load(f)
    apply_l5_payload(stream, payload)
    return True

def _index_keyed_to_uid(table: dict, attrs) -> dict:
    """Convert a {index: ...} table written before uids existed to {uid: ...}."""
    out = {}
    for k, v in (table or {}).items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if not (0 <= i < len(attrs)):
            continue                      # the region is gone; drop the row
        out[str(int(attrs[i].uid))] = v
    return out


def _migrate_symbol_tables(gr: dict, sm: dict, attrs) -> None:
    """Rewrite index-keyed reference tables in place. See `symbol_ids`."""
    if gr:
        gr["counts"] = _index_keyed_to_uid(gr.get("counts") or {}, attrs)
    if sm:
        rows = {}
        for k, row in (sm.get("counts") or {}).items():
            try:
                i = int(k)
            except (TypeError, ValueError):
                continue
            if not (0 <= i < len(attrs)):
                continue
            rows[str(int(attrs[i].uid))] = _index_keyed_to_uid(row or {}, attrs)
        sm["counts"] = rows
        sm["seen"] = _index_keyed_to_uid(sm.get("seen") or {}, attrs)


def _graph_index_to_uid(d: dict, attrs) -> None:
    """Rewrite a graph payload whose nodes are old table indices as uids.

    Only the endpoints change; `w_ab` is the weight leaving the stored `a`, so
    when a relabelling puts the two uids in the other order the two directions
    have to be swapped with them -- otherwise a -> b silently becomes b -> a.
    """
    rows = []
    for r in d.get("edges", []) or []:
        try:
            i, j = int(r["a"]), int(r["b"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= i < len(attrs) and 0 <= j < len(attrs)):
            continue                      # one end is gone; the edge is not
        ua, ub = int(attrs[i].uid), int(attrs[j].uid)
        if ua == ub:
            continue
        out = dict(r)
        if ua <= ub:
            out["a"], out["b"] = ua, ub
        else:
            out["a"], out["b"] = ub, ua
            out["w_ab"], out["w_ba"] = r.get("w_ba", 0.0), r.get("w_ab", 0.0)
            out["n_ab"], out["n_ba"] = r.get("n_ba", 0), r.get("n_ab", 0)
        rows.append(out)
    d["edges"] = rows
    ln = d.get("last_node")
    if ln is None:
        d["last_node"] = None
        return
    try:
        k = int(ln)
    except (TypeError, ValueError):
        d["last_node"] = None
        return
    d["last_node"] = int(attrs[k].uid) if 0 <= k < len(attrs) else None


def apply_l5_payload(stream, payload: dict) -> None:
    """Restore a stream from plain payload data (see build_l5_payload).

    The body below is the restore logic that used to live inside load_l5.
    It was moved, not rewritten: every backward-compatibility case here was
    paid for by a real incident and is kept verbatim.
    """
    from src.L5_cognition.terrain import Attractor
    from src.L5_cognition.dynamics import DynamicState
    ls = stream.landscape
    raw = payload.get("attractors")
    if raw is None:
        # Checkpoints written before the terminology change used "traces".
        raw = payload.get("traces", [])

    attrs = []
    for i, d in enumerate(raw):
        # Backward compatibility: records written by earlier versions may
        # lack fields. A missing decay factor is drawn from the same
        # distribution used at construction — but **deterministically**, from
        # (total_steps, index). Drawing it from `ls.state_space._rng` (unseeded)
        # meant two loads of the same archive produced two bodies that drift
        # apart: measured 2026-11-12, two projections of the same checkpoint
        # diverged at step 7 because their decay factors differed.
        df = d.get("_decay_factor", d.get("_sigma_decay"))
        if df is None:
            _rng = np.random.default_rng(int(payload.get("total_steps", 0) or 0) + i)
            df = float(1.0 + _rng.normal(0.0, ls.state_space.decay_spread))
            df = max(df, 0.05)
        a = Attractor(
            # **Copy, do not alias.** `build_l5_payload` stores `a.mu` itself,
            # not a copy, so `mu=d["mu"]` would make the loaded region share one
            # array with the archive it was loaded from -- and with every other
            # region loaded from the same payload. `Attractor.assign()` updates
            # the centroid in place (`self.mu += ...`), so a "read-only" sandbox
            # copy was writing into her: measured 2026-11-12, two projections of
            # the same checkpoint diverged at step 7 because they shared centroids.
            mu=np.array(d["mu"], dtype=float, copy=True), amp=d["amp"],
            scale=d.get("scale"), born_t=d.get("born_t"),
            _decay_factor=float(df),
        )
        a._assigned = int(d.get("_assigned", 0)) or len(d["members"])
        # Age structure (2026-11-09). Without this line every region comes back
        # with _born_step = 0, i.e. "born before the archive began", so age and
        # per-region lifetime are both wrong after every restart. Measured on
        # the real checkpoint (145504 steps, 307 regions): all 307 loaded with
        # _born_step = 0, so _population_visit_rate() returned 3.44e-05
        # (assigned / whole archive life) instead of the per-region rate. That
        # value is the numerator of _birth_depth(), so every region born after
        # the restart starts below the survival line (min_amp = 1e-3) and is
        # deleted by the next recover() -- the exact failure the 2026-11-07
        # audit fixed, reintroduced by restarting. Legacy checkpoints have no
        # such field; 0 is kept for them (age unknowable, treated as old).
        a._born_step = int(d.get("_born_step", 0))
        # When she was last here. 0 = unknown (legacy record); fold_auto treats
        # unknown as "cannot be shown to be overdue" rather than as "long ago".
        a._last_seen_step = int(d.get("_last_seen_step", 0))
        # Legacy records have no uid; 0 means "assign below", after the whole
        # table is built (the number depends on the order of this loop).
        a.uid = int(d.get("uid", 0) or 0)
        # Members must keep their bounded buffer. An earlier version passed
        # list(...) into the constructor; because the field had been assigned
        # explicitly it silently became an unbounded list, so every load
        # round trip removed the bound. One checkpoint ended up carrying
        # 68743 historical states for a single attractor, which inflated its
        # measured sigma with old outliers and never decayed.
        a.members = collections.deque(
            (np.array(m, dtype=float, copy=True) for m in list(d["members"])[-MEMBER_BUFFER:]),
            maxlen=MEMBER_BUFFER,
        )
        attrs.append(a)
    # Fill in uids for records that predate them, and never hand out a number
    # that is already in use: max(uid) and the stored counter decide what is
    # next.
    _next = int(payload.get("next_uid", 0) or 0)
    for _a in attrs:
        if int(_a.uid) <= 0:
            _a.uid = _next if _next > 0 else 1
            _next = int(_a.uid) + 1
        _next = max(_next, int(_a.uid) + 1)
    ls.state_space._next_uid = max(_next, 1)
    ls.state_space._attractors = attrs
    # Folded summaries. Absent from older checkpoints (nothing had been folded
    # yet when the field was introduced), so a missing key means "none", not
    # "damaged".
    ls.state_space._folded = [dict(f) for f in (payload.get("folded") or [])]
    gs = payload.get("global_scale")
    if gs is not None:
        ls.state_space._global_scale = float(gs)  # legacy field, unused
    # Do not trust a scale carried over from an older checkpoint.
    # Earlier versions could collapse it to ~1e-4 while the state was
    # oscillating in place, and every attractor created with that scale
    # was too small to grow. The scale is re-derived from observed motion
    # instead: the seeding branch runs on the next observation, and the
    # recent window only supplies a starting radius when nothing better
    # is available.
    ls.state_space._scale_seeded = False
    ls.state_space._last_x = None
    ls.state_space._center = None

    # L3 connectivity. An earlier version of this function did not store the
    # graph at all, so every restart discarded everything L3 had learned; the
    # edge count was observed to drop from 157 to 0 across a round trip.
    g = payload.get("graph")
    if g is not None:
        from src.L3_memory import TraceGraph
        if payload.get("graph_ids") != "uid":
            _graph_index_to_uid(g, attrs)
        stream.graph = TraceGraph.from_dict(g)

    # M1: the self-model and the error it makes. Missing in older checkpoints;
    # a fresh model is the honest starting point for those, since the counts
    # it would need were never recorded.
    sm = payload.get("self_model")
    gr = payload.get("grounding")
    if payload.get("symbol_ids") != "uid":
        _migrate_symbol_tables(gr if isinstance(gr, dict) else {},
                               sm if isinstance(sm, dict) else {}, attrs)
    if sm is not None:
        from src.L5_cognition.self_model import OnlineSelfModel
        stream.self_model = OnlineSelfModel.from_dict(sm)
    sg = payload.get("self_gain") or {}
    if sg:
        stream.self_gain.error = float(sg.get("error", stream.self_gain.error))
        stream.self_gain.observations = int(sg.get("observations", 0))
        stream.self_gain.hits = int(sg.get("hits", 0))
    ls_symbol = payload.get("last_symbol")
    stream._last_symbol = None if ls_symbol is None else int(ls_symbol)

    # L6 grounding. Missing in older checkpoints: every symbol starts as
    # "none", which is the honest state for a table that was never measured.
    if gr is not None:
        from src.L5_cognition.grounding import GroundingTable
        stream.grounding = GroundingTable.from_dict(gr)

    ident = payload.get("identity")
    if ident is not None:
        stream.identity._rounds = int(ident.get("rounds", 0))
        appear = {}
        for k, c in ident.get("appear", {}).items():
            a, b = k.split(",")
            appear[(int(a), int(b))] = int(c)
        stream.identity._appear.clear()
        stream.identity._appear.update(appear)
    ls.state_space._recent.extend(payload["recent"])
    ls._zero = payload["zero"]
    if payload["x"] is not None:
        stream._st = DynamicState(x=payload["x"], v=payload["v"])
    stream._step_count = payload["step_count"]
    ls.state_space._total_steps = int(payload.get("total_steps", 0)) or stream._step_count

    # Apply the scale floor immediately on load.
    #
    # Waiting until the first observation is too late: the checkpoint can
    # contain a region whose measured scale is far below anything the space
    # currently uses, and if the state happens to be sitting on it, the very
    # first gradient evaluation uses the unclamped scale. Measured: one load
    # produced a gradient of 1471 and drove speed to 14.7 in a single step,
    # before the floor was applied at the end of that same step.
    ls.state_space._apply_scale_floor()
    ls.state_space._scale_floor_pending = False


def build_acc_payload(accumulator) -> dict:
    """The slow accumulator: her pressure level, which must survive the trip."""
    return {"P": accumulator.P, "hist": list(accumulator._hist)}


def save_acc(accumulator, name: str = "acc.pkl") -> Path:
    """慢累积量：她的"压力"水平，跨会话必须延续。"""
    d = _ensure()
    p = d / name
    _atomic_dump(build_acc_payload(accumulator), p)
    return p


def load_acc(accumulator, name: str = "acc.pkl") -> bool:
    p = _ensure() / name
    if not p.exists():
        return False
    with p.open("rb") as f:
        payload = pickle.load(f)
    accumulator.P = payload["P"]
    accumulator._hist.extend(payload["hist"])
    return True


# ============================================================
# 批量
# ============================================================

def snapshot_all(objs: dict[str, Any]) -> dict[str, str]:
    """
    一次性保存全部形状。

    objs 形如 {"l0": net, "l1": predictor, "l2": field, "acc": accumulator}
    或含 "l5": stream。
    """
    saved = {}
    if "l0" in objs:
        saved["l0"] = str(save_l0(objs["l0"]))
    if "l1" in objs:
        saved["l1"] = str(save_l1(objs["l1"]))
    if "l2" in objs:
        saved["l2"] = str(save_l2(objs["l2"]))
    if "acc" in objs:
        saved["acc"] = str(save_acc(objs["acc"]))
    if "l5" in objs:
        saved["l5"] = str(save_l5(objs["l5"]))

    # 盖时间章：coma.json 是唯一的时间权威（见 src/common/coma.py）。
    #
    # 放在 finally 里：某一次落盘失败不该让 saved_at 停在很久以前。
    # 否则下一次启动会把"她其实一直在跑的那段时间"算成昏迷 —— 实测过这个后果：
    # save_l1 抛错期间，gap 从 2.7 s 一路涨到 8.9 s，涨的就是进程醒着的时长。
    # 语义因此定为"最后一次落盘尝试的时刻"，是 gap 的下界，宁可少算昏迷。
    try:
        if "l5" in objs:
            saved["l5"] = str(save_l5(objs["l5"]))
    finally:
        try:
            from src.common import coma as _coma
            _coma.mark_saved()
        except Exception as e:
            print(f"[persist] coma 时间章失败（不影响状态保存）: {e}")

    return saved


def restore_all(objs: dict[str, Any]) -> dict[str, bool]:
    """一次性恢复全部形状。返回每个组件是否成功。"""
    out = {}
    if "l0" in objs:
        out["l0"] = load_l0(objs["l0"])
    if "l1" in objs:
        out["l1"] = load_l1(objs["l1"])
    if "l2" in objs:
        out["l2"] = load_l2(objs["l2"])
    if "acc" in objs:
        out["acc"] = load_acc(objs["acc"])
    if "l5" in objs:
        out["l5"] = load_l5(objs["l5"])
    return out
