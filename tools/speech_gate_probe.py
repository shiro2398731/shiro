r"""「说」的闸只读探针（2026-11-28）。

P4 离线，但 SpeechGate 的逻辑可以单独跑：只用副本的 transition 序列，
看闸本来会放行多少句、压掉多少句、为什么压。

Run: .\.venv\Scripts\python.exe tools\speech_gate_probe.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402
from src.L6_expression.gate import SpeechGate  # noqa: E402

STEPS = int(os.environ.get("SHIRO_STEPS", "500"))
CP = Path("data/state/_audit_speech.pkl")


def main() -> int:
    shutil.copyfile(Path("data/state/l5.pkl"), CP)
    st = Stream()
    assert persist.load_l5(st, name=CP.name)
    gate = SpeechGate()
    prev_uid = None
    transitions = 0
    reasons = Counter()
    # 2026-11-28：对照——「说」由关系（张力最大 pair）触发，而不是区域切换触发
    said_pairs = deque(maxlen=64)
    rel_speak = 0
    rel_events = 0
    prev_pair = None
    for _ in range(STEPS):
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        # 区域切换门（现状）
        uid = st.detector.current_attractor
        if uid is not None and uid != prev_uid:
            transitions += 1
            gate.note_transition(int(uid))
            gate.should_speak()
            reasons[gate.last_reason] += 1
        prev_uid = uid
        # 关系门（对照）：最大张力 pair 变化且没说过
        ts = sorted(st.drive.last, key=lambda t: t.amount, reverse=True)
        pair = (int(ts[0].a), int(ts[0].b)) if ts else None
        if pair is not None and pair != prev_pair:
            rel_events += 1
            if pair not in said_pairs:
                said_pairs.append(pair)
                rel_speak += 1
        prev_pair = pair
    try:
        CP.unlink()
    except FileNotFoundError:
        pass
    out = {
        "steps": STEPS,
        "transitions": transitions,
        "would_speak": gate.utterances,
        "suppressed": gate.suppressed,
        "speak_share_of_transitions": round(gate.utterances / max(transitions, 1), 3),
        "reasons": dict(reasons),
        "said_pairs_memory": gate.said_count,
        "relation_events": rel_events,
        "relation_would_speak": rel_speak,
    }
    dest = Path("data/exp/speech_gate_probe.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("== 说的闸（副本 %d 步）==" % STEPS)
    for k, v in out.items():
        print(f"  {k}: {v}")
    print(f"  明细: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


