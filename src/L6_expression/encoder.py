"""
src/L6_expression/encoder.py

Input path: natural language -> signal the rest of the system can carry.

This is a codec, not an interpretation. It does one thing: turn a sentence into
a fixed-length numeric vector plus a small set of discrete tags, so that the
layers above can consume it the same way they consume anything else from
outside.

What this is not
----------------

It does not decide what the sentence means. It does not produce a summary, a
sentiment judgement, or a response. If the model used here were removed, the
output would be a different number for the same sentence, and nothing else in
the system would be affected -- that is the test of whether something is a
codec or a judgement.

The encoder's only freedom is the projection from text to numbers. Everything
downstream treats those numbers as input, not as conclusions.

Two modes
---------

`LlmEncoder`
    Calls an LLM with a strict instruction: return a JSON object containing a
    vector and tags, no prose. Validated on return; malformed output is
    rejected rather than repaired.

`HashEncoder`
    Deterministic, no network. Turns text into a vector by hashing character
    n-grams. Weaker, but it always works and it is reproducible: the same
    sentence always produces the same vector, which matters for testing and for
    verifying that the pipeline is wired correctly.

Both implement the same interface, so the rest of the system does not care which
one is in use.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datetime import datetime, timezone


@dataclass
class TextSignal:
    """
    What the input path produces.

    vector : (D,) float array in [-1, 1], the numeric form of the sentence
    tags   : discrete labels; only structural fields, no semantic verdict
    text   : the original sentence, kept for logging only
    t_event: timestamp
    """
    vector: np.ndarray
    tags: dict = field(default_factory=dict)
    text: str = ""
    t_event: str = ""

    def to_dict(self) -> dict:
        return {
            "vector": [float(x) for x in self.vector],
            "tags": dict(self.tags),
            "text": self.text,
            "t_event": self.t_event,
        }


# Fields the LLM is allowed to produce. Anything else is discarded.
_ALLOWED_TAGS = {"length", "question", "negation", "imperative"}


class HashEncoder:
    """
    Deterministic text -> vector, no network.

    Method: character n-grams hashed into D buckets with signed accumulation,
    then L2-normalized and scaled to [-1, 1] by dimension.

    Properties that matter here:
        - same text always gives the same vector
        - different texts rarely collide (n-grams of length 3 over the
          full character set)
        - no model, no API, no cost
    """

    def __init__(self, dim: int = 128, ngram: int = 3):
        self.D = int(dim)
        self.n = int(ngram)
        self._last_error: Optional[str] = None

    def available(self) -> bool:
        return True

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def encode(self, text: str) -> TextSignal:
        s = (text or "").strip()
        v = np.zeros(self.D, dtype=float)
        if s:
            for i in range(max(1, len(s) - self.n + 1)):
                gram = s[i:i + self.n]
                h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.D
                sign = 1.0 if h[4] & 1 else -1.0
                v[idx] += sign

        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm

        # scale into [-1, 1] so downstream sees a bounded input
        v = np.clip(v * np.sqrt(self.D), -1.0, 1.0)

        tags = {
            "length": "short" if len(s) <= 8 else ("mid" if len(s) <= 32 else "long"),
            "question": bool(s.endswith(("?", "？", "吗", "么"))),
            "negation": any(k in s for k in ("不", "没", "别", "未")),
            "imperative": bool(s and s[-1] in "，。!！"),
        }
        return TextSignal(
            vector=v, tags=tags, text=s,
            t_event=datetime.now(timezone.utc).isoformat(),
        )


class LlmEncoder:
    """
    text -> vector via an LLM constrained to emit JSON only.

    The instruction admits no prose and no analysis; the response is parsed and
    validated. A response that does not conform is rejected, not repaired.
    """

    BASE_URL = "https://api.deepseek.com/chat/completions"
    MODEL = "deepseek-chat"
    TIMEOUT_S = 20.0

    PROMPT = (
        "You are an encoder, not an assistant. Convert the given sentence into "
        "a numeric vector of exactly {dim} values, each between -1 and 1, that "
        "represents the sentence's form (its characters, rhythm, and structure). "
        "Do not summarise. Do not interpret. Do not answer. "
        "Reply with JSON only, no prose, in this exact shape: "
        '{{"vector": [...{dim} numbers...], "tags": {{"length": "short|mid|long", '
        '"question": true|false, "negation": true|false, "imperative": true|false}}}}'
    )

    def __init__(self, api_key: str, dim: int = 128):
        self._api_key = api_key
        self.D = int(dim)
        self._last_error: Optional[str] = None

    @classmethod
    def from_env(cls, env_var: str = "DEEPSEEK_API_KEY", dim: int = 128):
        k = os.environ.get(env_var)
        if not k:
            return None
        return cls(k, dim=dim)

    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def encode(self, text: str) -> Optional[TextSignal]:
        s = (text or "").strip()
        if not s:
            return None

        body = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": self.PROMPT.format(dim=self.D)},
                {"role": "user", "content": s},
            ],
            "stream": False,
            "temperature": 0.0,
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.BASE_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_S) as resp:
                raw = resp.read().decode("utf-8")
            obj = json.loads(raw)
            content = obj["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            self._last_error = f"HTTP {e.code}"
            return None
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            return None

        parsed = self._parse(content)
        if parsed is None:
            return None
        vec, tags = parsed
        return TextSignal(
            vector=vec, tags=tags, text=s,
            t_event=datetime.now(timezone.utc).isoformat(),
        )

    def _parse(self, content: str) -> Optional[tuple[np.ndarray, dict]]:
        """Validate strictly. Malformed output is rejected, never repaired."""
        t = content.strip()
        if t.startswith("```"):
            t = t.strip("`")
            if t.startswith("json"):
                t = t[4:]
        try:
            d = json.loads(t)
        except json.JSONDecodeError:
            self._last_error = "not json"
            return None

        vec = d.get("vector")
        if not isinstance(vec, list) or len(vec) != self.D:
            self._last_error = f"vector length {len(vec) if isinstance(vec, list) else 'n/a'}"
            return None
        try:
            v = np.array([float(x) for x in vec], dtype=float)
        except (TypeError, ValueError):
            self._last_error = "non-numeric vector"
            return None
        if not np.all(np.isfinite(v)):
            self._last_error = "non-finite vector"
            return None

        raw_tags = d.get("tags") or {}
        tags = {k: raw_tags[k] for k in _ALLOWED_TAGS if k in raw_tags}
        return np.clip(v, -1.0, 1.0), tags


def make_encoder(dim: int = 128):
    """
    Always return the deterministic encoder.

    LlmEncoder is kept in this module but is not selected here, because it was
    measured and does not translate. On three identical sentences it produced
    vectors of norm 0.847, 0.862, 0.836 with pairwise distances of 1.098 and
    1.201 -- a cosine similarity near 0.1, i.e. essentially unrelated
    directions. On a different sentence the norm was 7.04, eight times larger.

    A model cannot count strokes or measure rhythm; asked to map a sentence to
    numbers it produces plausible-looking noise. An encoder whose output
    changes when nothing changed cannot support learning: the same word heard
    twice would land in different places each time.

    The hash encoder is deterministic by construction, which is the property
    that matters here. It is a weaker representation, and that is honest.

    Set SHIRO_ENCODER=llm to force the model path for comparison; the fact that
    it is reachable does not mean it should be used.
    """
    if os.environ.get("SHIRO_ENCODER", "").lower() == "llm":
        enc = LlmEncoder.from_env(dim=dim)
        if enc is not None and enc.available():
            return enc, "llm"
    return HashEncoder(dim=dim), "hash"
