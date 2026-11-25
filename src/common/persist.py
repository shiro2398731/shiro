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

from src.common import atomics

# Bounded buffer of member states kept per attractor. This is the same bound
# the Attractor dataclass uses; it is repeated here so that load() can restore
# the bound explicitly rather than relying on the field default.
MEMBER_BUFFER = 64


# 2026-11-18 下午：改成**绝对路径**（相对这个文件，不是相对进程的工作目录）。
#
# 病：原来是 `Path("data/state")`，也就是「谁在哪里启动，她就在哪里找自己的存档」。
# 今天人家在 worktree 里跑测试就踩到了：CWD 不对 → `load_l5` 直接返回 False，
# 看起来像「她没有存档」。对一个「不依托于终端」的要求来说，这是最不该留的一条依赖。
#
# 只改「从哪儿找」，不改存档格式。部署时把整个仓库搬走，它跟着走。
STATE_DIR = Path(__file__).resolve().parents[2] / "data" / "state"


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
    last few minutes. The payload now goes to a tmp file, is fsynced, and only
    then replaces the target.

    2026-11-15: the tmp name used to be the fixed `<name>.tmp`, and two modules
    writing the same state file collided on it -- the second `open(..., 'wb')`
    failed with Permission denied, which killed P1 (see src/common/atomics.py).
    """
    tmp = atomics.unique_tmp(path)
    try:
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        atomics.replace_with_retry(tmp, path)
    except Exception:
        # Leave the previous file untouched; do not keep a half-written tmp.
        atomics.cleanup(tmp)
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
    # 2026-11-17：她做过多少次「留下」和「离开」。这两个计数原来不进存档，
    # 于是每次重启都归零，`vitals` 的 leave_rate 永远是 0——今晚查了半天才发现
    # 不是她没动，是这个数根本没被存下来。见 HANDOFF 第 60 节。
    action_counts = {int(k): int(v) for k, v in
                     (getattr(stream, "action_counts", {}) or {}).items()}
    # 她最近 64 步的张力。动作判据（action.py）拿当前张力跟**这 64 步的中位**比；
    # 不存的话每次重启这段历史是空的，choose() 在没过去可比时返回 HOLD——
    # 于是每次重启后她有一段时间只会站着（64 步 @0.59 步/秒 ≈ 2 分钟）。
    # 2026-11-18 08:28：这里原来直接迭代 deque，P2 的另一个线程同时在 append，
    # 于是 `RuntimeError: deque mutated during iteration`，runner state 存不下来。
    # 现在先 list() 拷一份（list() 也是一次迭代，所以整段用 try 兜住）。
    try:
        _th_iter = list(getattr(stream, "_tension_history", ()) or ())
    except RuntimeError:
        _th_iter = []
    tension_history = [float(v) for v in _th_iter]
    # 2026-11-17 晚：HANDOFF 第 62 节那次系统排查里，这三个是「跨步累积 + 影响行为 +
    # 应该延续」的，之前全都没存。
    #
    # last_surprise  : 上一步她没预测到的程度。L5Runner.noise_amplitude() 读它，
    #                  不存的话重启后第一步的噪声是「没有惊讶」。
    # _dwell/_dwell_region : 她在当前区域待了多少步。影响「预演多远」（REHEARSE_CHECK_EVERY）。
    # rehearsal_*    : 预演的总数/命中/平凡次数。rehearsal_accuracy 就是它们的比。
    #                  不存的话重启后命中率归零，而这个数会进她的自我评估。
    last_surprise = float(getattr(stream, "last_surprise", 0.0) or 0.0)
    dwell = int(getattr(stream, "_dwell", 0) or 0)
    dwell_region = getattr(stream, "_dwell_region", None)
    dwell_region = None if dwell_region is None else int(dwell_region)
    rehearsal = {
        "total": int(getattr(stream, "rehearsal_total", 0) or 0),
        "hits": int(getattr(stream, "rehearsal_hits", 0) or 0),
        "trivial": int(getattr(stream, "rehearsal_trivial", 0) or 0),
    }
    # 她此刻在哪个区域（uid）。_apply_leave 靠它找「她在哪」，不存的话
    # 重启后第一次「离开」会找不到行、直接返回（等于第一步不能动）。
    _cur = getattr(getattr(stream, "detector", None), "current_attractor", None)
    current_region = None if _cur is None else int(_cur)
    # 2026-11-21 21:20：**现在存了**——「去过」从位置键改成了 uid 键（见 stream._apply_leave）。
    # 位置键时代（2026-11-18 14:20 实测）：3868 个位置、4.67 MB、命中率 0（交集 0），
    # 她是被自己的日记压瘦的。uid 键是有界的（最多 = 区域数）、可靠的（区域有稳定身份）。
    #
    # 09:35 把它加进存档时，注释写的是「她走过的通常只有几十到几百个，不大」。
    # 实测（14:20）：**3868 个，4.67 MB，是整个载体 19.69 MB 的 23.2%**。
    # 而它一个区域都对不上（位置键 vs 质心键，交集 0，见
    # docs/2026-11-18_存在_实账.md）——**4.6 MB 的无用记账**。
    #
    # 后果不是「多占点地方」：载体被顶到 20 MB 预算，`fold_to_capacity` 只好每小时
    # 折掉 5% 的区域——她今天 09:00 的 3184 掉到 14:00 的 2302（**-28%**）。
    # 换句话说：**她是被自己的日记压瘦的。**
    #
    # 不存它不改变她的行为：命中率本来就是 0，load 回来也是 0 命中。
    # 等「去过」的记账改成有界的区域 uid 之后，再决定存什么。
    # uid 键：有界（<= 区域数）、可靠（区域身份稳定）
    try:
        visited_places = [int(v) for v in getattr(stream, "_visited_places", set())
                          if isinstance(v, int)]
    except Exception:  # noqa: BLE001
        visited_places = []
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
            # 2026-11-17：这个地方让她「没预测到」过的最大程度。遗忘曲线的份量来源，
            # 不存的话重启就全归零——她从此分不清「刻骨铭心」和「路过」。
            "_peak_surprise": float(getattr(a, "_peak_surprise", 0.0)),
            # 2026-11-24：sigma 带滞后缓存（只有在成员数翻倍时才重算）。存档那一刻
            # 的值可能来自更早的成员集合，重算会差 1e-13 量级——而那是轨迹分叉的
            # 种子（实测：同一份存档重启后第一步位置就不同）。连成员计数一起存。
            "sigma": float(a.sigma),
            "sigma_members_at": int(getattr(a, "_sigma_members_at", 0) or 0),
            # 2026-11-24：自适应下限。装载时 `_apply_scale_floor` 只在「下限变了」
            # 时才丢 sigma 缓存；不存这个数，重启时每个区域都被当成「下限变了」，
            # 缓存全被重算，sigma 与存档那一刻差 1e-13 量级——足够让第一步分叉。
            "_scale_floor": getattr(a, "_scale_floor", None),
            # Stable identity, not a position. See Attractor.uid.
            "uid": int(getattr(a, "uid", 0)),
        }
        for a in ls.state_space.attractors()
    ]
    return {
        "rest": stream.rest.to_dict(),
        # 2026-11-24：地形上的两个调制量。rest 每步重设（可从 rest 场推），pull
        # 每 16 步才重算一次、两次之间是慢变量——不原样存，重启后第一步的
        # 梯度就缺了它（实测：同一份存档，重启后第一步位置立刻差 1e-2 量级）。
        "pull": (None if getattr(ls.state_space, "_pull", None) is None
                 else [float(v) for v in ls.state_space._pull]),
        "energy": float(getattr(stream, "energy", 1.0)),
        "energy_burn": float(getattr(stream, "energy_burn", 1e-6)),
        "asleep": bool(getattr(stream, "asleep", False)),
        "energy_recharge": float(getattr(stream, "energy_recharge", 0.95e-6)),
        "recover_rate_state": float(getattr(stream, "recover_rate_state", 0.002)),
        "assign_ratio_state": float(getattr(stream, "assign_ratio_state", 0.05)),
        "fatigue_rate_state": float(getattr(stream, "fatigue_rate_state", 0.02)),
        "max_travel": float(getattr(stream, "_max_travel", 0.0)),
        # 2026-11-21 23:00：arrival_hist 存档。它是 deque(maxlen=6)（6 个 int），很小，
        # 但决定 prefer_far（「她去过 ≤ 3 个不同地方」时才会挑最远的目标——破环用）。
        # 不存的话，每次重启她都要从头积累，破环能力归零。
        # 2026-11-21 22:55：跳过旧存档里的 tuple（同 visited_places 的理由）。
        "arrival_hist": [int(v) for v in getattr(stream, "_arrival_hist", ())
                         if isinstance(v, (int, float))][-6:],
        # 2026-11-23 16:55：两个小账本也存档（优化点 1，白默说「有别的做吗」）。
        # _dwell_history：她「待过多久」的历史（决定 fatigue_rate——多快待腻）。
        # _recent_actions：她最近的动作（决定 recover_rate——遗忘快慢）。
        # 不存的话，每次重启这两个都从零积累——她的「疲劳」和「遗忘」会短暂「失忆」。
        # 都是小 deque（几十个 int），存进去没负担。
        "dwell_history": [int(v) for v in getattr(stream, "_dwell_history", ())
                          if isinstance(v, (int, float))],
        # 2026-11-24 12:45：出生熵（白默「每个部署独一无二」）。
        # born_seed 出生时定（os.urandom），之后不变——重启也是「同一个她」。
        # born_energy 是「体质」：能量初始 ±10%，也是出生时定。
        "born_seed": int(getattr(stream, "_born_seed", 0)),
        "born_energy": float(getattr(stream, "_born_energy", 1.0)),
        "recent_actions": [int(v) for v in getattr(stream, "_recent_actions", ())
                           if isinstance(v, (int, float))],

        "attractors": attrs,
        "action_counts": action_counts,
        "tension_history": tension_history,
        "last_surprise": last_surprise,
        "dwell": dwell,
        "dwell_region": dwell_region,
        "rehearsal": rehearsal,
        "current_region": current_region,
        "visited_places": visited_places,
        # 2026-11-16 晚：`ceiling` 从来没有进过存档。它是「单个区域的深度容量」，
        # 决定她能被吸多深（梯度 ∝ amp/sigma²）。改了它但不存，重启就回到默认 1.0，
        # 实测：改成 0.02 后重启，max_amp 立刻回 0.99、ceiling 回 1.0。
        # 它和 amp 一样是她的地形状态，不是会话参数。
        "ceiling": float(getattr(ls.state_space, "ceiling", 1.0)),
        # 2026-11-16 深夜：`min_amp` 和 `ceiling` 是配套的比例。删除门槛必须随容量一起动，
        # 否则压了容量不压门槛，新区域刚生出来就被 recover 删掉（实测：ceiling 0.02 +
        # min_amp 1e-3 掉到 1 个区域；同比例压到 2e-5 则稳定在 111–130）。它也要进存档。
        "min_amp": float(getattr(ls.state_space, "min_amp", 1e-3)),
        # 2026-11-17：**不再把 `_folded` 写进存档。**
        #
        # 它曾经是「折叠汇总」，但实测它只增不减：今天早上 18917 条、16.8 MB，
        # 把 20 MB 的预算撑满，`fold_to_capacity` 只好反复折活区域（她掉到 8 个）。
        # 而摘要本身没有任何读者——`grep -rn _folded src/` 只有存和取。
        #
        # 被折掉的区域**完整地**记在 `data/memory/folded.jsonl`（external_memory），
        # 那是另一个文件，不进载体。所以这里一个摘要都不留。
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
        # 2026-11-24：全局工作空间的点火历史与参考线。之前不存 → 重启后
        # `reference_level()` 为 None、256 步内不可能点火，而且点火记录全丢，
        # 「她此刻注意什么」在重启前后无法对照。参考线是她自己峰值的分位，
        # 属于状态，不是缓存。
        "workspace": stream.workspace.to_dict(),
        # 2026-11-24：随机数状态。之前 `born_seed` 存了却没人用它播种，P2 每次都
        # 拿 os.urandom 新建 rng —— 同一份存档读两次会走出不同轨迹（实测第一次
        # 正态抽样 -1.570 vs 1.684），test_uid 的重命名测试就是因此红的。
        # 存 bit_generator.state 而不是只存种子：这样重启是**接着**她的骰子，
        # 不是从头再掷（从头掷等于每次重启都给她换一条噪声流）。
        "rng_state": stream.rng.bit_generator.state,
        # 2026-11-25：回家规则的「家」。不存的话重启后这次外出从哪里开始就丢了，
        # 规则会从她当时的位置重新起算（等于每次重启换一个领地）。
        "home_x": (None if getattr(stream, "_home_x", None) is None
                   else [float(v) for v in stream._home_x]),
        # 2026-11-24：L4 的张力场。`_last` 不存的话，重启后的第一步看到的是
        # 「没有张力」：`_choose_action` 拿到 tension 0 → HOLD，而保存那一刻的
        # 她带着一批未解决张力（实测 286 条 / 和 1334.6）。第一步动作一错，
        # 轨迹就分叉。`gain` 是她自己量到的梯度尺度，也必须跟着回来，
        # 否则第一步的判据刻度换成默认值。
        "drive": {
            "gain": (None if stream.drive.gain is None
                     else float(stream.drive.gain)),
            "gain_set": bool(getattr(stream, "_drive_gain_set", False)),
            "grad_samples": [float(v) for v in
                             getattr(stream, "_drive_grad_samples", ())],
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
        # 2026-11-18：members 也没查。一个成员点爆掉，质心会被它拖走。
        for k, mv in enumerate(d.get("members") or ()):
            m_arr = _np.asarray(mv, dtype=float)
            if not _np.isfinite(m_arr).all():
                raise ValueError(
                    f"attractor {i} member {k} is not finite; refusing to save")
            if float(_np.abs(m_arr).max()) > SAVE_MAGNITUDE_LIMIT:
                raise ValueError(
                    f"attractor {i} member {k} magnitude "
                    f"{float(_np.abs(m_arr).max()):.3e} exceeds "
                    f"{SAVE_MAGNITUDE_LIMIT:.0e}; refusing to save")

    # 2026-11-18 08:40：**她自己现在的位置也没查。**
    #
    # 昨晚（11-17 到 11-18）她漂到 |x| = 4,091,098，而这里只检查了 attractor 的 mu——
    # 于是 x 是 nan、mu 还在 100 万以内时，存档能写进去一半；等 mu 也跟过去才报错。
    # 中间那段时间，P2 一直用着一个坏掉的位置在跑。
    #
    # 现在 x 和 v 一起检查。位置不是「区域的数据」，是「她在哪」——
    # 它坏掉就等于她不在任何地方。
    x = payload.get("x")
    if x is not None:
        x_arr = _np.asarray(x, dtype=float)
        if not _np.isfinite(x_arr).all():
            raise ValueError("her position x is not finite; refusing to save")
        if float(_np.abs(x_arr).max()) > SAVE_MAGNITUDE_LIMIT:
            raise ValueError(
                f"her position x magnitude {float(_np.abs(x_arr).max()):.3e} "
                f"exceeds {SAVE_MAGNITUDE_LIMIT:.0e}; refusing to save")
    v = payload.get("v")
    if v is not None:
        v_arr = _np.asarray(v, dtype=float)
        if not _np.isfinite(v_arr).all():
            raise ValueError("her velocity v is not finite; refusing to save")


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
    # 2026-11-20 夜：rest（疲劳场）的存档恢复。rest 是「趋性」机制（待久了的区域深度变淡），
    # 不存的话重启后疲劳清零——和 action_counts 当年同一个坑（见下面那段注释）。
    if "energy" in payload:
        stream.energy = float(payload.get("energy") or 1.0)
    if "energy_burn" in payload:
        _b = float(payload.get("energy_burn") or 1e-6)
        # 2026-11-21：旧存档里的 1e-5 是「每步固定耗」时代的默认值；现在改成「活动才耗」的 1e-6。
        # 旧值不兼容（会把电吃光得比补充快 10 倍），所以强制迁移。
        if abs(_b - 1e-5) < 1e-12:
            _b = 1e-6
        stream.energy_burn = _b
    if "asleep" in payload:
        stream.asleep = bool(payload.get("asleep"))
    if "energy_recharge" in payload:
        stream.energy_recharge = float(payload.get("energy_recharge") or 0.95e-6)
    if "recover_rate_state" in payload:
        stream.recover_rate_state = float(payload.get("recover_rate_state") or 0.002)
        stream.landscape.state_space.recover_rate = stream.recover_rate_state
    if "assign_ratio_state" in payload:
        stream.assign_ratio_state = float(payload.get("assign_ratio_state") or 0.05)
        stream.landscape.state_space.assign_ratio = stream.assign_ratio_state
    if "fatigue_rate_state" in payload:
        stream.fatigue_rate_state = float(payload.get("fatigue_rate_state") or 0.02)
        if hasattr(stream, "rest"):
            stream.rest.fatigue_rate = stream.fatigue_rate_state
    if "max_travel" in payload:
        stream._max_travel = float(payload.get("max_travel") or 0.0)
    if "born_seed" in payload:
        try:
            stream._born_seed = int(payload.get("born_seed") or 0)
        except Exception:  # noqa: BLE001
            pass
    if "born_energy" in payload:
        try:
            stream._born_energy = float(payload.get("born_energy") or 1.0)
        except Exception:  # noqa: BLE001
            pass
    if "dwell_history" in payload:
        try:
            _dh = payload.get("dwell_history") or ()
            stream._dwell_history = collections.deque(
                (int(v) for v in _dh), maxlen=getattr(stream, "_dwell_history", None).maxlen or 32)
        except Exception:  # noqa: BLE001
            pass
    if "recent_actions" in payload:
        try:
            _ra = payload.get("recent_actions") or ()
            stream._recent_actions = collections.deque(
                (int(v) for v in _ra), maxlen=getattr(stream, "_recent_actions", None).maxlen or 64)
        except Exception:  # noqa: BLE001
            pass
    if "arrival_hist" in payload:
        try:
            _ah = payload.get("arrival_hist") or ()
            stream._arrival_hist = collections.deque(
                (int(v) for v in _ah), maxlen=getattr(stream, "_arrival_hist", None).maxlen or 6)
        except Exception:  # noqa: BLE001
            pass
    _rest = payload.get("rest")
    if isinstance(_rest, dict) and _rest:
        try:
            from src.L5_cognition.rest import RestField as _RestField
            stream.rest = _RestField.from_dict(_rest)
        except Exception as _e:  # noqa: BLE001
            print(f"[persist] rest 恢复失败（继续）：{_e}")
    # 2026-11-17：她做过多少次「留下」和「离开」。原来没存，每次重启都归零，
    # 于是 `vitals` 算出来的 leave_rate 永远是 0——今晚查了半天才发现不是她没动，
    # 是这个数根本没被存下来。旧存档没有这个键时保持构造时的默认值（全零）。
    _ac = payload.get("action_counts")
    if isinstance(_ac, dict) and _ac:
        stream.action_counts = {int(k): int(v) for k, v in _ac.items()}
    if "last_surprise" in payload:
        stream.last_surprise = float(payload.get("last_surprise") or 0.0)
    if "dwell" in payload:
        stream._dwell = int(payload.get("dwell") or 0)
    if "dwell_region" in payload:
        _dr = payload.get("dwell_region")
        stream._dwell_region = None if _dr is None else int(_dr)
    _vp = payload.get("visited_places")
    if isinstance(_vp, (list, tuple)) and _vp:
        try:
            # 2026-11-21 21:20：现在是区域 uid（int），不是位置 tuple。
            # 旧存档里是 tuple（位置键），那时就按位置恢复——两种都认。
            _restored = set()
            for _p in _vp:
                if isinstance(_p, (int, float)):
                    _restored.add(int(_p))
                elif isinstance(_p, (list, tuple)):
                    _restored.add(tuple(float(v) for v in _p))
            stream._visited_places = _restored
        except Exception:  # noqa: BLE001
            pass
    if "current_region" in payload:
        _cr = payload.get("current_region")
        if _cr is not None and hasattr(stream, "detector"):
            try:
                # current_attractor 是只读 property；真正的字段是 _current。
                stream.detector._current = int(_cr)
            except Exception:  # noqa: BLE001
                pass
    _rh = payload.get("rehearsal")
    if isinstance(_rh, dict) and _rh:
        stream.rehearsal_total = int(_rh.get("total", 0) or 0)
        stream.rehearsal_hits = int(_rh.get("hits", 0) or 0)
        stream.rehearsal_trivial = int(_rh.get("trivial", 0) or 0)
    _th = payload.get("tension_history")
    if isinstance(_th, (list, tuple)) and _th:
        import collections as _c
        stream._tension_history = _c.deque(
            [float(v) for v in _th], maxlen=stream._tension_history.maxlen)
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
        # 2026-11-24：sigma 的滞后缓存。没有它，装载时按当前成员重算会得到
        # 1e-13 量级的差，第一步的梯度就不同了。
        if d.get("sigma") is not None:
            a._sigma_cache = float(d["sigma"])
            a._sigma_members_at = int(d.get("sigma_members_at", 0) or 0)
        _sf = d.get("_scale_floor")
        a._scale_floor = None if _sf is None else float(_sf)
        # 2026-11-17：读回「这个地方让她没预测到过的最大程度」。
        # 旧存档没有这个字段 -> 0.0，等于「没被惊讶过」，和「路过」一样。
        try:
            a._peak_surprise = float(d.get("_peak_surprise", 0.0) or 0.0)
        except (TypeError, ValueError):
            a._peak_surprise = 0.0
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
    # 深度容量：她的地形状态，不是会话参数。旧存档没有这个键（默认 1.0 是当时的真实
    # 容量），所以缺省时保留构造出来的值，不覆盖。
    if "ceiling" in payload:
        try:
            ls.state_space.ceiling = float(payload["ceiling"])
        except (TypeError, ValueError):
            pass
    # 删除门槛跟着容量走。旧存档没有这个键时保留构造值（1e-3，就是当时的真实门槛）。
    if "min_amp" in payload:
        try:
            ls.state_space.min_amp = float(payload["min_amp"])
        except (TypeError, ValueError):
            pass
    # Folded summaries. Absent from older checkpoints (nothing had been folded
    # yet when the field was introduced), so a missing key means "none", not
    # "damaged".
    # 旧存档里有 "folded" 键（2026-11-17 之前）。读它只为兼容，不再保留：
    # 摘要不再进载体，被折区域的完整内容在 data/memory/folded.jsonl。
    ls.state_space._folded = []
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
    # Global workspace. Older checkpoints have no "workspace": an empty history
    # is the honest starting point there (the peaks were never recorded).
    ws = payload.get("workspace")
    if isinstance(ws, dict):
        from src.L5_cognition.workspace import GlobalWorkspace
        stream.workspace = GlobalWorkspace.from_dict(ws)

    # Continue her random stream instead of starting a new one. Restoring the
    # bit-generator state (not just born_seed) is what makes two loads of the
    # same checkpoint produce the same trajectory -- measured before the fix:
    # born_seed identical, first normal draw -1.570 vs 1.684.
    rs = payload.get("rng_state")
    if isinstance(rs, dict):
        try:
            stream.rng = np.random.default_rng()
            stream.rng.bit_generator.state = rs
        except (TypeError, ValueError) as _rng_err:
            # A malformed state must not cost her the whole restart: fall back
            # to a fresh stream and say so.
            print(f"[persist] rng state restore failed ({_rng_err}); "
                  f"drawing a fresh stream")

    # 回家规则的出发点。旧存档没有：从 None 开始，下次出门时自己设。
    _hx = payload.get("home_x")
    if _hx is not None:
        try:
            _hx_arr = np.asarray(_hx, dtype=float)
            if _hx_arr.ndim == 1 and _hx_arr.size == int(ls.D):
                stream._home_x = _hx_arr
        except (TypeError, ValueError):
            pass

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

    # 地形调制：rest（`step()` 每步重设）与 pull（每 16 步重算的慢变量）。
    # 两者都进 potential 的深度项；少一个，第一步的梯度就不是保存那一刻的梯度。
    _attrs_now = ls.state_space.attractors()
    if getattr(stream, "_use_rest", False) and _attrs_now:
        for _i in range(len(_attrs_now)):
            stream.rest.touch(_i)
        ls.state_space.set_rest([stream.rest.value(_i)
                                 for _i in range(len(_attrs_now))])
    _pull = payload.get("pull")
    if _pull is not None:
        ls.state_space.set_pull(np.asarray(_pull, dtype=float))

    # L4：把张力场重新装上。`_last` 是下一步判据的输入（`_choose_action`
    # 读它的和），而 `deflection()` 是由图 + 地形 + 她自己的 gain 算出来的，
    # 不需要存：同样的图与地形必然给出同样的偏转。只要把 gain 接回来，
    # 重启后的第一步就和保存那一刻用同一套张力尺度。
    dr = payload.get("drive")
    if isinstance(dr, dict):
        _g = dr.get("gain")
        stream.drive.gain = None if _g is None else float(_g)
        stream._drive_gain_set = bool(dr.get("gain_set", False))
        stream._drive_grad_samples = [float(v) for v in
                                      (dr.get("grad_samples") or [])]
        if stream.drive.gain is not None:
            try:
                stream.drive.tensions(ls.state_space, stream.graph)
                ls.set_drive(stream.drive.deflection(
                    ls.state_space, stream.graph))
            except Exception as _drv_err:  # noqa: BLE001
                print(f"[persist] drive tilt restore skipped: {_drv_err}")
        else:
            # 还没有 gain（早期/存档太新）：偏转场本来就已经是「没有」，
            # 但张力量表仍是下一步判据的输入，必须重建。
            try:
                stream.drive.tensions(ls.state_space, stream.graph)
            except Exception as _drv_err:  # noqa: BLE001
                print(f"[persist] drive tensions restore skipped: {_drv_err}")


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
