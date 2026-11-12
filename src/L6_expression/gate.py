"""
src/L6_expression/gate.py

Whether she speaks at all. Not what she says.

What this file does not contain, on purpose
-------------------------------------------
1. **No text.** There is no string in it that could become a sentence, and the
   only output is a bool. If the gate could produce language, "the gate" would
   be the author speaking through her (C3, docs §11). `tests/test_speech_gate.py`
   fails if a sentence-shaped constant ever appears here.
2. **No pragmatics.** Its inputs are region ids and counters: no text, no
   directive, no "someone asked". INV-8 (docs §10) says pragmatics may not
   reach a decision; a gate that read the inbox would make "understanding"
   degrade into "obeying". The same test fails if this module imports anything
   from the project at all, which is how that stays true without review.
3. **No rate the author picked.** The criterion is "this transition has not
   already been said", measured against her own speech history. There is no
   sentence-per-minute number to tune.

Criterion
---------
    note_transition(region)   every L5 transition, in order
    should_speak()            True when (previous region, current region) has
                              not been spoken before, within her memory of
                              SPOKEN_MEMORY transitions

What that gives, and what the tests pin: a region she keeps circling is
described once; a transition she has not described is described; a transition
that carries no change is not described; and when her memory is full of
everything she has said, she stops -- the honest outcome, because there is
nothing new to report, not because a counter ran out.

Determinism
-----------
A bounded history and counters only: no RNG, no clock, no wall time. The same
transition sequence produces the same decisions, which is what lets a replay
reproduce her speech rather than approximate it.
"""

from __future__ import annotations

import collections
from typing import Optional

# Transitions remembered as "already said". A buffer size (C1), not a criterion:
# it bounds memory. When it is full, the oldest utterance stops counting as
# "already said" instead of silencing her permanently.
SPOKEN_MEMORY = 512


class SpeechGate:
    """Decides whether a transition becomes an utterance. Writes nothing."""

    def __init__(self, enabled: bool = True, memory: int = SPOKEN_MEMORY) -> None:
        self.enabled = bool(enabled)
        self.memory = int(memory)
        self._said: collections.deque = collections.deque(maxlen=self.memory)
        self._prev: Optional[int] = None
        self._cur: Optional[int] = None
        self.utterances = 0
        self.suppressed = 0
        self.last_reason = "no_transition"

    # ---------- input ----------

    def note_transition(self, region: Optional[int] = None) -> None:
        """One L5 transition. `region` is the region she moved into, if known."""
        if region is None:
            region = None
        else:
            try:
                region = int(region)
            except (TypeError, ValueError):
                region = None
        self._prev = self._cur
        self._cur = region

    # ---------- the one output ----------

    def should_speak(self) -> bool:
        """True only when there is a transition she has not already described."""
        if not self.enabled:
            return self._suppress("disabled")
        if self._cur is None or self._prev is None:
            return self._suppress("no_transition")
        if self._cur < 0:
            return self._suppress("no_region")
        if self._cur == self._prev:
            return self._suppress("no_change")
        pair = (self._prev, self._cur)
        if pair in self._said:
            return self._suppress("already_said")
        self._said.append(pair)
        self.utterances += 1
        self.last_reason = "new_transition"
        return True

    # ---------- record ----------

    def _suppress(self, reason: str) -> bool:
        self.suppressed += 1
        self.last_reason = reason
        return False

    @property
    def said_count(self) -> int:
        return len(self._said)
