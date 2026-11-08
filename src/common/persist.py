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

def save_l0(net, name: str = "l0.pkl") -> Path:
    """L0：神经元状态与连接。不含任何波形历史。"""
    d = _ensure()
    payload = {
        "E_v": net.E_v, "E_w": net.E_w,
        "I_v": net.I_v, "I_w": net.I_w,
        "mid_v": net.mid_v, "mid_w": net.mid_w,
        "slow_v": net.slow_v, "slow_w": net.slow_w,
        "EE": net.EE, "EI": net.EI, "IE": net.IE,
        "E_to_mid": net.E_to_mid, "mid_to_slow": net.mid_to_slow,
        "step_count": net.step_count,
        "envelope_A": getattr(net, "envelope_A", None),
    }
    p = d / name
    _atomic_dump(payload, p)
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
    payload = {
        "W_pred": predictor.W_pred,
        "prev_vec": getattr(predictor, "_prev_vec", None),
    }
    p = d / name
    _atomic_dump(payload, p)
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


def save_l2(field, name: str = "l2.pkl") -> Path:
    """
    L2：各维归一化历史、协方差样本、上一帧向量。
    这些决定"她怎么理解自己的状态"，比任何单帧数值都重要。
    """
    d = _ensure()
    payload = {
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
    p = d / name
    _atomic_dump(payload, p)
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
        }
        for a in ls.state_space.attractors()
    ]
    return {
        "attractors": attrs,
        "graph": stream.graph.to_dict(),
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
        "global_scale": ls.state_space.global_scale,
        "recent": list(ls.state_space._recent),
    }


def save_l5(stream, name: str = "l5.pkl") -> Path:
    """L5 checkpoint. See build_l5_payload for the contents."""
    d = _ensure()
    p = d / name
    _atomic_dump(build_l5_payload(stream), p)
    return p


def load_l5(stream, name: str = "l5.pkl") -> bool:
    p = _ensure() / name
    if not p.exists():
        return False
    with p.open("rb") as f:
        payload = pickle.load(f)
    apply_l5_payload(stream, payload)
    return True

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
    for d in raw:
        # Backward compatibility: records written by earlier versions may
        # lack fields. A missing decay factor is drawn from the same
        # distribution used at construction.
        df = d.get("_decay_factor", d.get("_sigma_decay"))
        if df is None:
            df = float(1.0 + ls.state_space._rng.normal(
                0.0, ls.state_space.decay_spread))
            df = max(df, 0.05)
        a = Attractor(
            mu=d["mu"], amp=d["amp"],
            scale=d.get("scale"), born_t=d.get("born_t"),
            _decay_factor=float(df),
        )
        a._assigned = int(d.get("_assigned", 0)) or len(d["members"])
        # Members must keep their bounded buffer. An earlier version passed
        # list(...) into the constructor; because the field had been assigned
        # explicitly it silently became an unbounded list, so every load
        # round trip removed the bound. One checkpoint ended up carrying
        # 68743 historical states for a single attractor, which inflated its
        # measured sigma with old outliers and never decayed.
        a.members = collections.deque(
            list(d["members"])[-MEMBER_BUFFER:], maxlen=MEMBER_BUFFER
        )
        attrs.append(a)
    ls.state_space._attractors = attrs
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
        stream.graph = TraceGraph.from_dict(g)

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


def save_acc(accumulator, name: str = "acc.pkl") -> Path:
    """慢累积量：她的"压力"水平，跨会话必须延续。"""
    d = _ensure()
    payload = {"P": accumulator.P, "hist": list(accumulator._hist)}
    p = d / name
    _atomic_dump(payload, p)
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
