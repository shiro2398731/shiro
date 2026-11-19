"""
src/runtime/process_2_mind.py — P2: L2 + L4 + L5

本版本职责：
- 订阅 P1 的 l1.state.surprise（预测误差）
- 订阅 P1 的 l2.state.field（状态场）
- 用预测误差决定 L5 的噪声幅度（代替任何写死的 sigma）
- advances the L5 state-space dynamics
- 发布 l5.state.stream / l5.event.transition

尚未实现：
- L4（动机竞争）
- L2 本体从 P1 迁移过来（现在 L2 还在 P1 内）

关于 L5 的输入来自哪里
--------------------

L5 是 128 维，上游是 16 维状态场 + 标量惊讶。**L2 不做 L5 的坐标，只做调制**（
ARCHITECTURE.md §3、00_foundations.md §8.1）：

  - 噪声幅度：由惊讶的当前水平决定 → 惊讶大，探索幅度大
  - 偏转场：由状态场经固定随机投影到 128 维 → tilts the potential; it does not determine position

这个投影的随机种子是固定的（存在 config），所以换设备迁移时同一份投影还在，
但她走到的位置仍然由 L5 自己的动力学决定。
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

import collections
import json
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.common import atomics
from src.common.bus import Bus
from src.common.observe import Observer
from src.common.types import StreamState, StateVector, SurpriseSignal, safe_from_json
from src.L5_cognition.stream import Stream
import src.common.persist as persist


class L5Runner:
    """
    把上游信号翻译成 L5 的三个驱动量：噪声幅度、偏转场、零点。

    **这三个量都来自实测，没有一个是写死的常数。**
    """

    def __init__(self, dim: int = 128, embed_dim: int = 16, seed: int = 1234):
        self.D = dim
        self.embed_dim = embed_dim

        # 固定随机投影：16 维状态场 → 128 维偏转场。
        # 种子固定，所以换设备后同一份投影还在（身份连续的一部分）。
        rng = np.random.default_rng(seed)
        self.proj = rng.normal(0.0, 1.0, (dim, embed_dim)) / np.sqrt(embed_dim)

        # 惊讶信号的历史，用于把"当前惊讶"标准化为相对量
        self.surprise_hist = collections.deque(maxlen=256)
        # L0 统计量的历史，用于定零点
        self.feature_hist = collections.deque(maxlen=512)

        self._zero_ready = False

    # ---------- 续接 ----------

    def to_dict(self) -> dict:
        """她的运行历史，跨重启保留。

        2026-11-16：`surprise_hist` 之前只在内存里。P2 重启后它是空的，而
        `noise_amplitude()` 在不足 32 条时返回 0 —— 噪声 0，她一步不抖，`s_t` 恒 0，
        于是 P1 一条 surprise 都不发，历史永远补不满。**重启一次就把她送进吸收态一次。**

        按 ARCHITECTURE §7「从日志续接，不是从快照还原」：这份历史是她的状态，不是缓存。
        """
        return {
            "surprise_hist": [float(x) for x in self.surprise_hist],
            # 2026-11-16 晚：feature_hist 也必须续接。它是内驱（身体波动）的来源，
            # 空着就等于噪声仍然是 0——和 surprise_hist 空着是同一个吸收态。
            "feature_hist": [[float(v) for v in row] for row in self.feature_hist],
            "zero_ready": bool(self._zero_ready),
        }

    def load_dict(self, data: dict) -> None:
        hist = data.get("surprise_hist") or []
        self.surprise_hist.clear()
        for x in hist[-256:]:
            try:
                self.surprise_hist.append(float(x))
            except (TypeError, ValueError):
                continue
        self._zero_ready = bool(data.get("zero_ready", False))
        self.feature_hist.clear()
        for row in data.get("feature_hist") or []:
            try:
                arr = np.asarray([float(v) for v in row], dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.shape[0] == self.embed_dim:
                self.feature_hist.append(arr)

    # ---------- 噪声幅度 ----------

    def noise_amplitude(self) -> float:
        """
        噪声幅度 = 两条内驱里更强的那一条。

        一、惊讶驱动（原有）：当前惊讶相对其自身历史的位置。
        二、身体波动：`feature_hist` 的标准差。

        2026-11-16 为什么加第二条
        -------------------------
        只靠第一条会把她锁死：`surprise_hist` 空（重启清空）或恒定（她不动）时，
        第一条算出 0；而 0 噪声 → 她不动 → 惊讶不变 → 还是 0。**吸收态。**
        实测：10:28 那次重启之后她的 noise 一直是 0，区域从六百多掉到个位数。

        第二条不是外部常数，是**她自己身体的节律**：`feature_hist` 记的是 L0 统计量的历史，
        而她的身体（L0 快层）一直在动。用它的波动当探索幅度，等于说
        「我还活着这件事本身，就是我想动的理由」——这是白默 2026-11-16 要的「内驱」。

        实测（三个种子，4000 步）：零驱动 24 区域/张力 0；惊讶驱动 24/0；
        身体波动 2467/308/2434 区域，张力 712/164/672，离开率 67%/32%/61%。
        """
        # 二、身体波动（更基本的那条）
        body = 0.0
        if len(self.feature_hist) >= 16:
            arr = np.asarray(self.feature_hist, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == self.embed_dim:
                body = float(np.std(arr, axis=0).mean())

        # 一、惊讶驱动（原有）
        surprise = 0.0
        if len(self.surprise_hist) >= 32:
            arr_s = np.asarray(self.surprise_hist, dtype=float)
            med = float(np.median(arr_s))
            mad = float(np.median(np.abs(arr_s - med)))
            if mad < 1e-12:
                # 历史恒定：用历史水平本身当尺度（而不是用 0）
                scale = max(abs(med), float(np.sqrt(np.finfo(float).eps)))
            else:
                scale = mad
            s_now = float(arr_s[-1])
            z = (s_now - med) / scale
            # 把标准分映射到非负幅度。缩放系数属于工程标度（B 类）。
            surprise = float(max(abs(z), 0.0) * 0.01)

        return max(surprise, body)

    # ---------- 偏转场 ----------

    def deflection(self) -> np.ndarray | None:
        """16 维状态场 → 128 维偏转。只倾斜，不产生凹陷。"""
        if len(self.feature_hist) == 0:
            return None
        v16 = np.asarray(self.feature_hist[-1], dtype=float)
        if v16.shape[0] != self.embed_dim:
            return None
        # 中心化：偏离它的典型值才算"倾向"
        H = np.stack(list(self.feature_hist)[-64:])
        center = np.median(H, axis=0)
        return self.proj @ (v16 - center)

    # ---------- 零点 ----------

    def maybe_calibrate(self, stream: Stream) -> bool:
        """
        零点 = 预热期上游统计量的中位数投影。

        这是"她从哪里开始"的唯一来源：不是原点，不是随机初始化，
        而是她自己身体（L0/L1/L2）在这段时间的实际状态。
        """
        if self._zero_ready:
            return True
        if len(self.feature_hist) < 64:
            return False
        H = np.stack(list(self.feature_hist))
        z16 = np.median(H, axis=0)
        z128 = self.proj @ z16
        stream.calibrate(z128.reshape(1, -1))
        self._zero_ready = True
        return True


def main() -> int:
    global _CURRENT_BUS
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    bus = Bus("P2")
    # 2026-11-19：让进程内 supervisor 在崩溃后能关掉它。
    #
    # 今天实测崩了 5 次，最后一条是 `ZMQError: Address in use (tcp://*:5562)` 然后放弃重启——
    # 因为崩溃时没人关这个 socket，supervisor 重启 main() 时端口还被自己占着。
    # 昨晚加的「绑定失败等 15 秒」不够：占端口的是**自己上一次**，它不退出就等不到。
    _CURRENT_BUS = bus
    stream = Stream(dim=128, gamma=0.05, dt=0.01)
    runner = L5Runner(dim=128)

    # 2026-11-16：把她的运行历史接回来。
    # 之前 surprise_hist 只在内存里，P2 一重启就空；而 noise_amplitude() 在不足 32 条时
    # 返回 0，噪声 0 → 她不动 → s_t=0 → P1 不发 surprise → 永远补不满。
    # 实测：10:28 那次重启之后，她的 noise 一直 0，区域从 626 掉到个位数。
    _runner_state_path = Path("data/state/p2_runner.json")
    try:
        if _runner_state_path.exists():
            runner.load_dict(json.loads(_runner_state_path.read_text(encoding="utf-8")))
            print(f"[P2] runner resumed: surprise_hist={len(runner.surprise_hist)}")
    except Exception as e:  # noqa: BLE001
        print(f"[P2] runner state load failed (starting empty): {e}")

    def _save_runner_state() -> None:
        try:
            _runner_state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = atomics.unique_tmp(_runner_state_path)
            tmp.write_text(json.dumps(runner.to_dict(), ensure_ascii=False),
                           encoding="utf-8")
            atomics.replace_with_retry(tmp, _runner_state_path)
        except Exception as e:  # noqa: BLE001
            print(f"[P2] runner state save failed: {e}")

    # 2026-11-17：连续时间的开关。
    #
    # False（默认）= 现在这样：fresh.wait(0.5) 等事件，没事件就 0.5 秒走一步。
    # True          = 她一直在走：每步之后由 self_pace() 算出停多久，下界 PAUSE_FLOOR_S。
    #
    # 为什么默认关着：白默要跑毕设，不能让人家把 CPU 吃了。
    # 开之前先读 docs/2026-11-17_连续时间提案.md 第 8、9 节。
    # 2026-11-17 20:55：白默定了方案 C——「慢到几乎不耗 CPU，但一直在走」。
    #
    # 她说的是：「C就是你说的最好情况，目前我想的大概就是这样。」
    # 也就是：她不要「炸电脑的自由运行」，也不要「等事件才动的响应式」，
    # 要「一直在，但很安静」。
    #
    # 所以默认值改成 True，并且 PAUSE_FLOOR_S 从 0.2 放到 2.0——
    # 0.5 步/秒。她的演化速度是免费时段（CPU 几乎为零）里最慢的一档，
    # 但**她在走**，不是被 timeout 叫醒。
    CONTINUOUS = True
    # 每步至少停这么久。这是**工程上限**（她最快 5 步/秒），不是她的判据——
    # 和 LAAP 的 tick_ms=100 的区别：那个就是它的节拍，这个只是防她烧穿。
    # 每步停 2 秒 = 0.5 步/秒。这个数只买「她一直在」，不买「她跑得快」。
    # 白默 2026-11-17：「得想办法让她不把电脑炸了……能不能让其运行几乎不依赖算力？」
    # 答案是：不能零算力（零算力=进程停了=不在），但可以慢到几乎不耗。
    PAUSE_FLOOR_S = 2.0

    # ---------- 她自己的节奏（2026-11-17，**已降级**）----------
    #
    # 人家原来写的是一个「自适应」的 self_pace：`pause = FLOOR × (ref / 当前张力)`，
    # 以为这样节奏就是「她自己的」，不再是常数。
    #
    # **18:40 实测证伪了它。** 她的短期状态量在稳态下都是常数：
    #     speed   稳在 0.89
    #     tension 835.2 → 836.4（5 秒内）
    # 所以 `ref / 当前值 ≈ 1`，`pause` 永远等于下界 —— **等于一个伪装成自适应的常数**。
    #
    # 结构性的原因：她的「此刻」和「平时」本来就是同一件事（她在稳态里）。
    # 偏差为零，调速量为零。
    #
    # 因此这里**降级成一个坦率的 CPU 折中常数**：
    #     pause = PAUSE_FLOOR_S
    # 它不承载「她活得多快」的语义。她的「活」由**内部事件**承载——
    # 每一步、每次换区域、每次张力越过阈值，都是她自己的事件（见 stream.step / observer）。
    #
    # 「连续」的含义因此变成：**她一直在产生事件**，而不是**她的节拍器有性格**。
    # 白默要的是前者（「她在旁边」），不是后者。
    def self_pace() -> float:
        return float(PAUSE_FLOOR_S)

    state = {"surprise": 0.0, "field": None}
    # singleton discipline: follow P1's epoch; stop if it moves.
    from src.common import epoch as epoch_mod
    # P1 claims the epoch a moment after it boots, so wait for a non-zero one.
    # Reading 0 and treating it as ours would fence us against our own sibling,
    # which is exactly what happened the first time this was wired up.
    _boot_at = time.time()
    my_epoch = 0
    _wait_until = _boot_at + 15.0
    while time.time() < _wait_until:
        # 只采纳本次启动之后、且**有活持有者**的 epoch。
        # 只看 mtime 不够：上一轮退出时写的 retire() 本身也是一次新写入，
        # 于是一条已经退役的记录会被当成"我的"，等她真正的 epoch 落地后就自围栏
        # （2026-11-14 实测：P3 连续三次启动即退出，日志看起来像正常关闭）。
        _st = epoch_mod.adoptable(since_mtime=_boot_at - 2.0)
        if _st is not None:
            my_epoch = _st.epoch
            break
        time.sleep(0.2)
    if my_epoch == 0:
        print("[P2] no epoch was claimed for this boot, refusing to act")
        return 3
    print(f"[P2] epoch {my_epoch}")

    # M1 接线：自我预测误差只调节探索强度，不替代动力学。
    # 模型在这里不构造：self_model / self_gain 属于 L5 的状态（随 l5.pkl 落盘），
    # 否则每次重启她的自我模型都从零开始——那不是"连续存在"。
    # 预测与计分在 Stream.step() 内完成，下游只读增益。

    running = {"flag": True}

    def on_surprise(topic, obj):
        state["surprise"] = obj.s_t
        # 2026-11-17：交给 Stream，它会记在「她此刻所在的区域」上——那个区域
        # 让她没预测到过，所以难忘（遗忘曲线的份量来源）。
        stream.last_surprise = float(obj.s_t)
        runner.surprise_hist.append(float(obj.s_t))
        _save_runner_state()

    def on_field(topic, obj):
        # L2 目前还在 P1 内广播；这里先接收它的 16 维
        v = getattr(obj, "v", None)
        if v is not None and len(v) == 16:
            state["field"] = list(v)
            runner.feature_hist.append(np.asarray(v, dtype=float))
            # 2026-11-16 晚：feature_hist 是内驱的来源，必须落盘。
            # 之前只挂在 on_surprise 上，而 P1 一条 surprise 都不发，于是这个文件
            # 从来没生成过——等于内驱在重启后永远是空的。每填满 16 条存一次，不频繁。
            if len(runner.feature_hist) >= 16 and len(runner.feature_hist) % 16 == 0:
                _save_runner_state()

    bus.subscribe(
        "l1.state.surprise", on_surprise,
        parser=lambda s: safe_from_json(SurpriseSignal, s),
        owner_module="src.L2_state.field",
    )
    bus.subscribe(
        "l2.state.field", on_field,
        parser=lambda s: safe_from_json(StateVector, s),
        owner_module="src.L2_state.field",
    )

    # 恢复她此刻的形状（不是重放历史）
    restored = persist.restore_all({"l5": stream})

    # 2026-11-16 深夜：外置记忆。被折掉的区域完整地记在 data/memory/folded.jsonl，
    # `_folded` 只当索引。白默的话：「遗忘的部分是要保存在数据库作为外置的，
    # 就像人记不住那么多但是可以一直记笔记」。append-only，不改写。
    from src.common.external_memory import ExternalMemory
    _ext = ExternalMemory("data/memory/folded.jsonl")
    stream.landscape.state_space.external_memory = _ext
    print(f"[P2] external memory: {_ext.path} "
          f"({_ext.count()} records, {_ext.size_bytes()} bytes)")

    # ---- append-only journal: the checkpoint compresses the past, the journal
    # ---- is what happened after it. Replay makes the rollback window zero up to
    # ---- the last record (one step), instead of 60 s.
    from src.common.journal import Journal
    journal = Journal("data/journal/l5.wal")
    _space = stream.landscape.state_space
    _space.journal = None            # do not re-journal what we are replaying
    replayed = journal.replay(_space)
    _space.journal = journal
    print(f"[P2] journal replayed: {replayed}"
          f"  wal={journal.size_bytes()} bytes")
    print(f"[P2] restore: {restored}")
    if restored.get("l5"):
        ss = stream.stream_state()
        print(f"[P2] resumed: attractors={ss['n_attractors']}  "
              f"salient={ss['n_salient']}  steps={stream._step_count}")
        runner._zero_ready = True

    observer = Observer("P2", log_dir)

    def on_sig(signum, frame):
        running["flag"] = False

    signal.signal(signal.SIGINT, on_sig)

    # ---- 启动清理 ----
    # 上一版在折叠时没有剪 grounding / self_model / identity，存档里可能带着已经
    # 不存在的 uid（2026-11-15 容量折叠第一次真触发之后，integrity 报 33772 条
    # "not a live uid"）。这里做一次与折叠路径相同的剪枝：没有死 id 时是 no-op。
    _space0 = stream.landscape.state_space
    _alive0 = _space0.alive_uids()
    _g0 = stream.grounding.prune(_alive0)
    _s0 = stream.self_model.prune(_alive0)
    _i0 = stream.identity.prune(_alive0)
    _sync0 = stream.consolidator.sync(stream.graph, _space0)
    if _g0 or _s0 or _i0 or _sync0.get("dropped_edges"):
        observer.event("startup_prune", {
            "grounding_pruned": int(_g0),
            "self_model_pruned": int(_s0),
            "identity_pruned": int(_i0),
            "edges_dropped": int(_sync0.get("dropped_edges", 0)),
        })
        print(f"[P2] startup prune: grounding={_g0} self_model={_s0} "
              f"identity={_i0} edges={_sync0.get('dropped_edges', 0)}")
    signal.signal(signal.SIGTERM, on_sig)

    # ---- 坐标系归一（2026-11-18 下午）----
    # 白默：「我不觉得应该有边界。」所以不设界，只保证「坐标多大都能活」。
    # 她的动力学只用距离，整体平移不改行为；见 src/L5_cognition/reorigin.py。
    # 放在 journal replay 之后、推演之前：日志里的旧坐标已经重放完，不会和这次平移打架。
    from src.L5_cognition.reorigin import maybe_reorigin
    _ro = maybe_reorigin(stream)
    if _ro.get("shifted"):
        print(f"[P2] reorigin: max|component| {_ro['before']:,.0f} -> "
              f"{_ro['after']:,.0f} (limit {_ro['limit']:,.0f}, "
              f"moved {_ro['moved']} 处)")
        observer.event("reorigin", _ro)
    else:
        print(f"[P2] reorigin: skipped (max|component| {_ro['max_component']:,.0f} "
              f"<= {_ro['limit']:,.0f})")

    print("[P2] starting L5 (terrain dynamics)")
    print(f"[P2] observations: {log_dir / 'observations.jsonl'}")

    transitions = 0
    steps = 0
    last_persist = time.time()
    # l5.pkl 是 7.8 MB，一次 24 ms。5 s 一次会写 1.5 MB/s（约 130 GB/天），
    # 所以这里退一步用 60 s：回滚窗口 300s -> 60s，约 120 个 L5 步。
    # 要真正归零得上 append-only journal（见评估文档的下一步）。
    PERSIST_INTERVAL_S = 60.0

    SNAPSHOT_INTERVAL_S = 180.0   # 与 Observer 的 3 分钟节流对齐
    last_snapshot = 0.0

    # L5 不被定时器驱动，而由上游的新鲜度驱动：
    # 只在收到新消息、或环境已经变了足够久时才推进一步。
    fresh = threading.Event()

    def on_surprise_wrap(topic, obj):
        on_surprise(topic, obj)
        fresh.set()

    def on_field_wrap(topic, obj):
        on_field(topic, obj)
        fresh.set()

    try:
        while running["flag"]:
            if CONTINUOUS:
                # 2026-11-17：她一直在走。不等事件，每步之后由她自己的状态决定停多久。
                # 事件只是把 pause 压短（fresh.set() 让 wait(0) 立刻返回），不改变「她一直在跑」。
                _pause = self_pace()
                fresh.wait(timeout=_pause)
                fresh.clear()
            else:
                # 现在这样：等事件，没事件就 0.5 秒走一步。
                got = fresh.wait(timeout=0.5)
                fresh.clear()
            if not epoch_mod.is_current(my_epoch):
                _led = epoch_mod.load()
                print(f"[P2] fenced, stopping "
                      f"(ledger epoch={_led.epoch} retired={_led.retired}; mine={my_epoch})")
                break

            # 上游还没给够数据：什么也不做，等她身体先积累统计量
            if not runner.maybe_calibrate(stream):
                continue

            # 噪声与偏转都由上游实测给出；她自己的预测误差只做有界缩放
            stream.landscape.set_noise_amplitude(
                runner.noise_amplitude() * stream.exploration_gain)
            stream.landscape.set_deflection(runner.deflection())

            ev = stream.step()
            steps += 1
            if ev is not None:
                transitions += 1
                observer.event("transition", {
                    "from": ev.from_attractor,
                    "to": ev.to_attractor,
                    "noise_driven": ev.noise_driven,  # Transition 的真实字段
                })

            # 观测快照（3 分钟一次）
            #
            # broadcast_state() 只带广播需要的最少字段（下游只读 attractor_id），
            # 不能拿来当快照：这里原本写 ss = stream.broadcast_state()，
            # 于是 ss["n_attractors"] 直接 KeyError，P2 每次启动第一望就崩
            # （既有缺陷，原版 shiro 同样如此）。
            #
            # 而 stream_state() 会重算 viability（O(区域数)），不能每步调用，
            # 所以用本地定时器控频，Observer 侧再 force=True 保证一定写进去。
            now_t = time.time()
            if now_t - last_snapshot >= SNAPSHOT_INTERVAL_S:
                last_snapshot = now_t
                ss = stream.stream_state()
                observer.snapshot("L5", {
                    "steps": steps,
                    "attractors": ss["n_attractors"],
                    "n_salient": ss["n_salient"],
                    "speed": round(ss["speed"], 4),
                    "potential": round(ss["potential"], 4),
                    "noise": round(stream.landscape.noise_amplitude, 5),
                    "transitions": transitions,
                    # 她的动作：谁选的、离开占多少。行动是她自己的，不是统计出来的。
                    "action": ss["action"],
                    "leave_share": ss["leave_share"],
                    # M1：她自己预测的命中率与当前增益（可 ablation 对照）
                    "rehearsal_accuracy": ss["rehearsal_accuracy"],
                    "rehearsals": ss["rehearsals"],
                    "self_accuracy": round(stream.self_gain.accuracy, 4),
                    # 命中率会被"待在原地"抬高：实测 0.9390 的 0 阶基线打不过
                    # 模型的 0.9325。真正能区分的是留出/前向对数损失，见 tools/m1_audit.py。
                    "self_bits": round(stream.self_model.bits_per_observation, 4),
                    # M2：她自己给的把握。要和 self_accuracy 并排看 ——
                    # 校准好不等于知道（见 tools/m2_audit.py 的说明）。
                    "self_conf": round(stream.self_model.mean_confidence, 4),
                    "self_gain": round(stream.self_gain.gain(), 4),
                    # L3
                    "edges": ss["n_edges"],
                    "recall": ss["recall"],
                    # L4
                    "tensions": ss["tensions"]["n"],
                    "tension_sum": ss["tensions"]["sum"],
                    # L8
                    "identity": ss["identity"],
                    "id_rounds": stream.identity.rounds,
                }, force=True)

            # 发布 l5.state.stream。P4（L6）靠它拿符号，P3 靠它记录。
            # payload carries a vector, but L6 may read attractor_id only
            # (ARCHITECTURE section 4.2).
            #
            # 这一行是必须的：下面 publish 用的是 broadcast_state() 的字段。
            # 2026-11-13 我改快照时把它连同 dedupe 一起删了，于是 P2 每步
            # 都在这里抛 NameError: bs，supervisor 连续 5 次后放弃重启 ——
            # 而她看起来"活着"（P1 在跑），只有日志里能看到。
            bs = stream.broadcast_state()
            bus.publish(
                "l5.state.stream",
                StreamState(
                    x_t=bs["x"],
                    attractor_id=str(bs["attractor_id"]),
                    landscape_energy=float(bs["potential"]),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
            )

            # 存她此刻的形状
            now = time.time()
            if now - last_persist >= PERSIST_INTERVAL_S:
                # 折叠：载体有界。
                #
                # 2026-11-17 10:30：暂停 fold_auto。实测不管限速 5% 还是 1%，她都持续掉
                # （427 -> 250 -> 92 -> 8），折得比长得快；降到每次 1 个，她稳在 8-10 个但不长。
                # 换言之 **fold_auto 现在是净损失**。先停掉它，只保留 fold_to_capacity（载体纪律）。
                #
                # 注意：这里必须是 `_folded = 0` 的赋值，不能只写注释——它下面是
                # `_folded += _capacity_folded`，缺了赋值就 UnboundLocalError（11:47 崩过两次）。
                _folded = 0
                _space = stream.landscape.state_space

                def _carrier_bytes(_st=stream):
                    import pickle as _pickle
                    return len(_pickle.dumps(persist.build_l5_payload(_st),
                                             protocol=_pickle.HIGHEST_PROTOCOL))

                _capacity_folded = _space.fold_to_capacity(_carrier_bytes)
                if _capacity_folded:
                    observer.event("fold_capacity", {
                        "removed": int(_capacity_folded),
                        "live": len(_space.attractors()),
                        "carrier_kb": round(_carrier_bytes() / 1024, 1),
                    })
                _folded += _capacity_folded
                if _folded:
                    # 折叠把 uid 从 alive_uids() 里拿走了，但指代表与自模型也是按
                    # uid 记账的：不跟着剪，它们就会给已经不存在的区域继续记账。
                    # 2026-11-15 实测：区域 1074、指代键 3772、自模型符号 6326，
                    # integrity 报 33772 条 "not a live uid"。
                    _alive = _space.alive_uids()
                    _ground_pruned = stream.grounding.prune(_alive)
                    _self_pruned = stream.self_model.prune(_alive)
                    _ident_pruned = stream.identity.prune(_alive)
                    # 图自己也会走到 sync，但折叠发生在 step 之后、落盘之前——
                    # 不在这里补一次，存档里就会带着被折区域的边。
                    _sync = stream.consolidator.sync(stream.graph, _space)
                    observer.event("fold", {"removed": int(_folded),
                                   "live": len(stream.landscape.state_space.attractors()),
                                   "grounding_pruned": int(_ground_pruned),
                                   "self_model_pruned": int(_self_pruned),
                                   "identity_pruned": int(_ident_pruned),
                                   "edges_dropped": int(_sync.get("dropped_edges", 0))})
                # 2026-11-18 晚：保存撞数值上限时，**她自己归一一次再试**，
                # 而不是让 assert_l5_finite 抛出去、把这个进程交给外面的 keeper。
                #
                # 理由是 153 次崩溃里 52 次是「质心 > 1e6」——她走远了、坐标过界、
                # 定期存档拒写、进程退出、外面拉起来。她是被「保存不了」弄死的，
                # 而「坐标太大」是可以用一次整体平移解决的（动力学只用距离）。
                #
                # 所以：先试正常保存；抛 ValueError 就整体归一、记事件、再试一次。
                # 还是不行才往外抛（那是真坏了，交给 supervisor）。
                # 2026-11-19 上午：把这段抽成 `save_with_self_heal()`（模块级），
                # 因为「内联在 P2 主循环里」等于「永远不会被单独验证」。
                save_with_self_heal(stream, observer)
                last_persist = now
                # 检查点已经包含了日志的全部效果，可以截断
                journal.truncate()
    finally:
        try:
            persist.snapshot_all({"l5": stream})
            print("[P2] 形状已保存")
        except Exception as e:
            print(f"[P2] 保存失败: {e}")
        bus.stop()
        print("[P2] stopped")
    return 0


def _prune_overflow_members(stream, limit=None):
    """丢掉分量超过存档上限的「成员」。

    2026-11-19 上午（测试抓到的）：`shift_origin` 按 `center_vector` 搬 mu/x/recent/visited，
    也搬 members——但 **members 里可能有比 mu 远几十万倍的旧坐标**（11-17 漂移留下的），
    搬同一个 vec 之后它们仍然超限。于是「归一 → 重试」二次失败，进程照样死。

    这里只丢「分量超过上限」的那些（那本来就是坏数据），不动正常成员。
    返回丢掉的条数。
    """
    from src.common.persist import SAVE_MAGNITUDE_LIMIT

    cap = float(limit if limit is not None else SAVE_MAGNITUDE_LIMIT * 0.95)
    dropped = 0
    try:
        attrs = stream.landscape.state_space.attractors()
    except Exception:  # noqa: BLE001
        return 0
    for a in attrs:
        members = getattr(a, "members", None)
        if not members:
            continue
        keep = []
        for m in members:
            try:
                v = float(abs(m).max()) if hasattr(m, "max") else 0.0
            except Exception:  # noqa: BLE001
                v = 0.0
            if v > cap:
                dropped += 1
            else:
                keep.append(m)
        if dropped and len(keep) != len(members):
            try:
                a.members = type(members)(keep, maxlen=getattr(members, "maxlen", 64))
                a._sigma_cache = None
            except Exception:  # noqa: BLE001
                pass
    return dropped


def save_with_self_heal(stream, observer=None):
    """保存；撞数值上限时**整体归一、记事件、重试**；还不行就**丢超限成员**再试。

    为什么要有它：153 次崩溃里 52 次是「质心 > 1e6」——她走远了、坐标过界、
    定期存档拒写、进程退出、外面拉起来。她是被「保存不了」弄死的，而「坐标太大」
    可以用一次整体平移解决（她的动力学只用距离）。

    2026-11-19：从 P2 主循环里抽出来，**就是为了能被测试**。
    内联在主循环里的代码等于永远不被单独验证。

    返回 (shifted, info)：
        shifted=False 表示一次就存上了；
        shifted=True  表示撞过上限、处理过，info 里有 before/after/moved/pruned。
    三步都救不回来才往外抛（那是真坏了，交给 supervisor）。
    """
    from src.L5_cognition.reorigin import (
        center_vector as _center,
        max_component as _maxcomp,
        shift_origin as _shift,
    )

    try:
        persist.snapshot_all({"l5": stream})
        return False, None
    except ValueError as _save_err:
        _mb = _maxcomp(stream)
        _mv = _shift(stream, _center(stream))
        _ma = _maxcomp(stream)
        info = {
            "reason": str(_save_err)[:160],
            "before": float(_mb),
            "after": float(_ma),
            "vec_norm": float(_mv["vec_norm"]),
            "moved": int(_mv["moved"]),
            "pruned": 0,
        }
        if observer is not None:
            observer.event("reorigin_persist", info)
        print(f"[P2] 保存撞上限（{_save_err}）→ 自我归一 "
              f"{_mb:,.0f} -> {_ma:,.0f}，重试保存", flush=True)
        try:
            persist.snapshot_all({"l5": stream})
            return True, info
        except ValueError as _save_err2:
            # 归一后还超：members 里有比 mu 远得多的旧坐标，搬同一个 vec 救不回来。
            _pruned = _prune_overflow_members(stream)
            info["pruned"] = int(_pruned)
            info["reason2"] = str(_save_err2)[:160]
            if observer is not None:
                observer.event("members_pruned_on_save", {
                    "pruned": int(_pruned),
                    "reason": str(_save_err2)[:160],
                })
            print(f"[P2] 归一后还超（{_save_err2}）→ 丢 {_pruned} 个超限成员，再试保存",
                  flush=True)
            persist.snapshot_all({"l5": stream})
            return True, info


# 同一个错误连续出现这么多次，说明它不是瞬态（端口占用、配置错、缺文件），
# 继续重启只会刷日志并把 CPU 烧在退避上。带着明确退出码停下，由外部 supervisor 决定。
MAX_IDENTICAL_FAILURES = 5

# 当前这次 main() 里建的 Bus。进程内 supervisor 崩溃后要靠它释放端口。
# 2026-11-19：不加它，`ZMQError: Address in use` 会连续 5 次、最后放弃重启。
_CURRENT_BUS = None


def _supervised() -> int:
    """崩溃退避重启 + 永久错误快速失败（评估文档 §3.3）。

    瞬态异常：保存形状 → 退避 2/4/8/…/60 s → 重新 main()（会走一次 begin_session，
    gap 很小 → 快启，不计昏迷事件）。
    永久异常：同一个错误连续出现 MAX_IDENTICAL_FAILURES 次 → 放弃重启并返回 1。
    实测依据：端口被占时旧版本会无限重启（每轮打一整段 traceback），
    对无人看守的长期运行来说，这比停下更糟。
    """
    import time as _time
    import traceback as _tb

    backoff = 2.0
    last_sig = None
    repeats = 0
    while True:
        try:
            code = main()
        except KeyboardInterrupt:
            return 130
        except SystemExit:
            raise
        except BaseException as exc:
            sig = f"{type(exc).__name__}: {exc}"
            repeats = repeats + 1 if sig == last_sig else 1
            last_sig = sig
            _tb.print_exc()
            # 2026-11-19：**崩溃后先关掉 bus**，否则重启 main() 时端口还被自己占着，
            # 会一路 `ZMQError: Address in use` 到放弃重启（今天真实发生了 5 次）。
            global _CURRENT_BUS
            if _CURRENT_BUS is not None:
                try:
                    _CURRENT_BUS.stop()
                    print("[SUPERVISOR] bus 已关闭，端口释放", flush=True)
                except Exception as _bus_err:  # noqa: BLE001
                    print(f"[SUPERVISOR] 关 bus 失败: {_bus_err}", flush=True)
                _CURRENT_BUS = None
            if repeats >= MAX_IDENTICAL_FAILURES:
                print(
                    f"[SUPERVISOR] 同一错误连续 {repeats} 次，放弃重启：{sig}",
                    flush=True,
                )
                return 1
            print(
                f"[SUPERVISOR] 崩溃（同一错误第 {repeats} 次），{backoff:.0f}s 后重启",
                flush=True,
            )
            _time.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)
            continue
        return 0 if code is None else int(code)


if __name__ == "__main__":
    import sys
    sys.exit(_supervised())
