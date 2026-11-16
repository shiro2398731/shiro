"""
src/runtime/process_3_life.py — P3: L3 + L7 + L8 (currently L7 only)

Responsibilities (L7 is complete; L3/L8 not yet implemented):
- Witness the L0 envelope phase (L7.rhythm.PhaseWitness)
- Record developmental stage (L7.stages.StageMonitor) - record only, never control
- Homeostasis monitoring (homeostasis.Homeostasis) - observe only, never touch
- Numeric safety: NaN/Inf may be repaired, because that is the compute environment,
  not her thinking

Not done, by design:
- no phase broadcast (the phase source is L0)
- no awake/asleep definition
- no "nudge her awake" (that would be interference)
- no time-based state switching

L3 / L8 are not implemented.
"""

from __future__ import annotations

import collections
import json
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.common import atomics
from src.common.bus import Bus
from src.common.homeostasis import Homeostasis
from src.common.observe import Observer
from src.common.types import EnvelopeEvent, StateVector, StreamState, SurpriseSignal, safe_from_json
from src.L7_lifecycle.rhythm import PhaseWitness
from src.L7_lifecycle.stages import StageMonitor

# How many region ids the recurrence/dwell statistics look at. A window size is
# an engineering quantity (C1). 64 is short enough to react within a snapshot
# interval (3 min) and long enough that a run of identical ids means something.
ID_WINDOW = 64
# Fewest ids before recurrence and dwell are worth computing at all.
MIN_IDS = 16


def main() -> int:
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    bus = Bus("P3")
    witness = PhaseWitness()
    # 阶段跨重启延续（2026-11-16）。之前每次 P3 新建 StageMonitor、_current 从 0 起，
    # events 里于是出现 4 -> 0；ARCHITECTURE §8 要求转换单向、没有退行。
    # 阶段只被记录、不控制任何层，所以这里只是把「她走到哪了」这个事实留住。
    _stage_path = Path("data/state/stage.json")
    stages = StageMonitor()
    try:
        if _stage_path.exists():
            stages = StageMonitor.from_dict(
                json.loads(_stage_path.read_text(encoding="utf-8")))
            print(f"[P3] stage resumed: {stages.current} ({stages.stage_def().name})")
    except Exception as e:  # noqa: BLE001
        print(f"[P3] stage load failed, starting from 0: {e}")
        stages = StageMonitor()

    def _save_stage() -> None:
        try:
            _stage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = atomics.unique_tmp(_stage_path)
            tmp.write_text(json.dumps(stages.to_dict(), ensure_ascii=False),
                           encoding="utf-8")
            atomics.replace_with_retry(tmp, _stage_path)
        except Exception as e:  # noqa: BLE001
            print(f"[P3] stage save failed: {e}")

    # 启动就落一次盘：否则一个停在 stage 0 的会话不会写文件，下次重启还是"没有记录"。
    _save_stage()

    homeo = Homeostasis()
    observer = Observer("P3", log_dir)

    stop = threading.Event()
    latest = {"dev": None, "s_avg": None, "pressure": None, "fast": None}

    def on_envelope(topic: str, obj: EnvelopeEvent) -> None:
        witness.see(obj.phase, obj.period_s, obj.t_event)
        observer.snapshot("envelope", {
            "phase": obj.phase,
            "period_s": round(obj.period_s, 2) if obj.period_s else None,
            "events": witness.count,
            "switch_rate_hz": (round(witness.switching_rate(), 3)
                               if witness.switching_rate() else None),
        })

    def on_field(topic: str, obj: StateVector) -> None:
        latest["dev"] = obj.dev
        homeo.observe(
            pressure=latest["pressure"] if latest["pressure"] is not None else 0.0,
            activity=latest["fast"] if latest["fast"] is not None else 0.0,
            dev=obj.dev,
        )
        stages.observe(dev=obj.dev)

    # ---- what the stage criteria need, and only what exists on the bus ----
    #
    # P3 used to feed StageMonitor `dev` alone, so three of the four criteria
    # had empty buffers and she could never leave stage 0. The two signals
    # below come from what she already broadcasts: prediction error from P1,
    # and the attractor id sequence from P2 (recurrence + dwell computed here,
    # so no layer has to know what the stages mean).

    def on_surprise(topic: str, obj: SurpriseSignal) -> None:
        stages.observe(s_avg=obj.s_t)

    ids: collections.deque = collections.deque(maxlen=ID_WINDOW)

    def on_stream(topic: str, obj: StreamState) -> None:
        try:
            aid = int(obj.attractor_id)
        except (TypeError, ValueError):
            return
        if aid < 0:
            return
        ids.append(aid)
        if len(ids) < MIN_IDS:
            return
        seq = list(ids)
        seen: set[int] = set()
        repeats = 0
        for x in seq:
            if x in seen:
                repeats += 1
            else:
                seen.add(x)
        runs = []
        run = 1
        for a, b in zip(seq, seq[1:]):
            if a == b:
                run += 1
            else:
                runs.append(run)
                run = 1
        runs.append(run)
        stages.observe(recurrence=repeats / float(len(seq)),
                       dwell=float(np.median(runs)))

    bus.subscribe(
        "l0.event.envelope", on_envelope,
        parser=lambda s: safe_from_json(EnvelopeEvent, s),
        owner_module="src.L7_lifecycle.rhythm",
    )
    bus.subscribe(
        "l2.state.field", on_field,
        parser=lambda s: safe_from_json(StateVector, s),
        owner_module="src.L7_lifecycle.stages",
    )
    bus.subscribe(
        "l1.state.surprise", on_surprise,
        parser=lambda s: safe_from_json(SurpriseSignal, s),
        owner_module="src.L7_lifecycle.stages",
    )
    bus.subscribe(
        "l5.state.stream", on_stream,
        parser=lambda s: safe_from_json(StreamState, s),
        owner_module="src.L7_lifecycle.stages",
    )

    def on_sig(signum, frame):
        stop.set()

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
        print("[P3] no epoch was claimed for this boot, refusing to act")
        return 3
    print(f"[P3] epoch {my_epoch}")

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    print("[P3] L7 见证中：相位 / 阶段 / 稳态（只看，不碰）")
    print(f"[P3] observations: {log_dir / 'observations.jsonl'}")

    last_eval = 0.0
    try:
        while not stop.is_set():
            stop.wait(1.0)
            if not epoch_mod.is_current(my_epoch):
                _led = epoch_mod.load()
                print(f"[P3] fenced, stopping "
                      f"(ledger epoch={_led.epoch} retired={_led.retired}; mine={my_epoch})")
                break
            now = time.time()
            if now - last_eval < 60:
                continue
            last_eval = now

            t_event = datetime.now(timezone.utc).isoformat()

            # 阶段：纯记录
            prev = stages.current
            idx = stages.evaluate(t_event)
            if idx != prev:
                observer.event("stage_advanced", {
                    "from": prev, "to": idx,
                    "name": stages.stage_def().name,
                })
                _save_stage()

            # 闭环：只记录
            rep = homeo.closure()
            if not rep.closed:
                observer.event("closure_broken", {
                    "pressure": round(rep.pressure_level, 4),
                    "activity": round(rep.activity_level, 4),
                    "corr": round(rep.correlation, 3),
                    "note": rep.note,
                })

            flat, why = homeo.is_flat()
            if flat:
                observer.event("state_flat", {"note": why})

            observer.snapshot("lifecycle", {
                "stage": stages.stage_def().name,
                "stage_index": idx,
                "stage_criteria": dict(stages.last_reasons),
                "closure_ok": rep.closed,
                "corr": round(rep.correlation, 3) if rep.correlation else None,
                "envelope_events": witness.count,
            })
    finally:
        bus.stop()
        print("[P3] stopped")
    return 0


# 同一个错误连续出现这么多次，说明它不是瞬态（端口占用、配置错、缺文件），
# 继续重启只会刷日志并把 CPU 烧在退避上。带着明确退出码停下，由外部 supervisor 决定。
MAX_IDENTICAL_FAILURES = 5


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
