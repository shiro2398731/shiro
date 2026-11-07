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

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.common.bus import Bus
from src.common.homeostasis import Homeostasis
from src.common.observe import Observer
from src.common.types import EnvelopeEvent, StateVector, safe_from_json
from src.L7_lifecycle.rhythm import PhaseWitness
from src.L7_lifecycle.stages import StageMonitor


def main() -> int:
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    bus = Bus("P3")
    witness = PhaseWitness()
    stages = StageMonitor()
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

    def on_sig(signum, frame):
        stop.set()

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    print("[P3] L7 见证中：相位 / 阶段 / 稳态（只看，不碰）")
    print(f"[P3] observations: {log_dir / 'observations.jsonl'}")

    last_eval = 0.0
    try:
        while not stop.is_set():
            stop.wait(1.0)
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
                "closure_ok": rep.closed,
                "corr": round(rep.correlation, 3) if rep.correlation else None,
                "envelope_events": witness.count,
            })
    finally:
        bus.stop()
        print("[P3] stopped")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
