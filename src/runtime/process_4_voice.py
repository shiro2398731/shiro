"""
src/runtime/process_4_voice.py

P4: the language interface. Two directions, both through one codec.

Output
------
L5 symbols are read from `l5.state.stream`; when a transition happens, the
symbols are handed to the LLM adapter, which writes a sentence to
data/output/utterances-YYYYMMDD.jsonl. The adapter returns None by design, so
nothing it produces can flow back into cognition.

Input
-----
Lines typed into this process (or appended to data/input/inbox.txt by any other
means) are encoded into a numeric signal and published on `l6.text.signal`.
P1 consumes that topic and feeds it into the predictor chain.

The two directions are deliberately separate. Speaking does not alter the state
that generates speech; hearing does not depend on whether speech occurred.

Encoding
--------
Text -> vector is done by src/L6_expression/encoder.py. With DEEPSEEK_API_KEY
present it uses the model constrained to emit JSON only; otherwise it falls back
to a deterministic hash encoder. Which one was used is recorded with every
signal, so a log cannot be misread as having used the model when it did not.

Why a file as well as stdin
---------------------------
Typing into a process that runs in the background is not always practical, and
the desktop client cannot write to stdin. Appending a line to inbox.txt is
equivalent and works from anywhere. The file is truncated after being read, so
each line is delivered once.
"""

from __future__ import annotations

import json
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

import signal
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.common.bus import Bus
from src.common.types import TextSignalEvent, safe_from_json
from src.L6_expression.llm_adapter import LlmAdapter, TemplateAdapter
from src.L6_expression.encoder import make_encoder

INBOX = Path("data/input/inbox.txt")
POLL_S = 0.5


def main() -> int:
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    INBOX.parent.mkdir(parents=True, exist_ok=True)

    import src.common.persist as persist
    from src.L5_cognition.stream import Stream

    stream = Stream(dim=128)
    if not persist.load_l5(stream):
        print("[P4] no L5 structure yet; start P2 first to build some")
    else:
        ss = stream.stream_state()
        print(f"[P4] read L5 structure: attractors={ss['n_attractors']} "
              f"salient={ss['n_salient']}")

    adapter = LlmAdapter.from_env(out_dir="data/output")
    if adapter is None:
        # C3: the template path is silent unless explicitly enabled, so without
        # a model she says nothing rather than something written by a person.
        print("[P4] no DEEPSEEK_API_KEY; template path is silent (origin=human forbidden)")
        adapter = TemplateAdapter(out_dir="data/output")
    else:
        print("[P4] output adapter ready (deepseek-chat)")

    encoder, enc_kind = make_encoder(dim=128)
    print(f"[P4] input encoder: {enc_kind}")

    bus = Bus("P4")
    from src.L6_expression.gate import SpeechGate
    gate = SpeechGate()
    stop = threading.Event()
    last_published = {"k": None}
    seen = {"n": 0}

    # ---------- output path ----------

    def on_stream(topic, obj):
        aid = getattr(obj, "attractor_id", -1)
        if aid == last_published["k"]:
            return
        last_published["k"] = aid
        # The region must be passed: without it the gate sees `None` for every
        # transition and `should_speak()` always answers "no_transition", so she
        # could never speak. Measured on a 2000-step id sequence (211 real
        # transitions): without the argument 0 utterances / 211 suppressed,
        # last_reason "no_transition" every time; with it 200 utterances / 11
        # suppressed.
        gate.note_transition(aid)
        if not gate.should_speak():
            return
        syms = {
            "trace": str(aid),
            "sequence": [str(aid)],
            "intent": "state" if aid >= 0 else "silent",
        }
        adapter.translate(syms, caller_note="placeholder")

    bus.subscribe(
        "l5.state.stream", on_stream,
        parser=lambda s: safe_from_json(_StreamView, s),
        owner_module="src.L6_expression.gate",
    )

    # ---------- input path ----------

    def deliver(text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        sig = encoder.encode(text)
        if sig is None:
            print(f"[P4] encoder rejected input ({(encoder.last_error or 'unknown')})")
            return
        bus.publish("l6.text.signal", TextSignalEvent(
            vector=[float(x) for x in sig.vector],
            tags=dict(sig.tags),
            text=sig.text,
            encoder=enc_kind,
            t_event=datetime.now(timezone.utc).isoformat(),
        ))
        seen["n"] += 1
        print(f"[P4] in ({enc_kind}) [{seen['n']}]: {sig.text[:40]}")

    def drain_inbox() -> None:
        """Read whole lines from the inbox and clear them."""
        if not INBOX.exists():
            return
        try:
            raw = INBOX.read_text(encoding="utf-8")
        except OSError:
            return
        if not raw.strip():
            return
        try:
            INBOX.write_text("", encoding="utf-8")
        except OSError:
            return
        for line in raw.splitlines():
            deliver(line)

    def stdin_loop() -> None:
        try:
            for line in _sys.stdin:
                if stop.is_set():
                    break
                deliver(line)
        except Exception:
            pass

    def on_sig(signum, frame):
        stop.set()

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    print("[P4] what is said here represents the current state, not a decision")
    print("[P4] output: data/output/utterances-*.jsonl")
    print(f"[P4] input : type here, or append lines to {INBOX}")

    t_stdin = threading.Thread(target=stdin_loop, daemon=True)
    t_stdin.start()

    try:
        while not stop.is_set():
            drain_inbox()
            stop.wait(POLL_S)
    finally:
        bus.stop()
        print("[P4] stopped")
    return 0


# Lightweight view: reads attractor_id from the L5 broadcast, never the vector
from dataclasses import dataclass


@dataclass
class _StreamView:
    """
    Reads attractor_id from the L5 broadcast only.

    x is parsed but used as a placeholder. The guard is on the LLM side: the
    whitelist in translate() drops x.
    """
    attractor_id: int = -1
    x: list = None

    @classmethod
    def from_json(cls, s: str) -> "_StreamView":
        d = json.loads(s)
        raw = d.get("attractor_id", "-1")
        try:
            aid = int(raw)
        except (TypeError, ValueError):
            aid = -1
        return cls(attractor_id=aid, x=d.get("x_t", []))


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
