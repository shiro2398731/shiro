"""
src/L6_expression/llm_adapter.py

LLM 只做语言区：把 L5 的离散符号翻译成自然语言。

红线（代码即法律）
------------------

1. **`translate()` 返回 None。** 它只做副作用（写文件），不返回任何东西给调用方。
   调用方不能等它、不能读它、不能拿它的结果去改任何认知层的状态。
   （ARCHITECTURE.md §4.3）

2. **输入白名单。** 只吃 SymbolBundle 里的离散量：痕迹 ID、跳盆序列、意图。
   任何向量、任何内部状态数值都不许传进来。

3. **prompt 是静态常量。** 不随系统状态变化，不包含"你应该""请考虑"这类
   会暗示推理方向的措辞。

4. **可以缺席。** 网络断了、API 报错、超时，全部降级为"说不出话"，
   而她继续运转。

**这个模块当前是占位实现**（白默 2026-11-04 明确要求）：
L6 现在说不代表"她想说"，只代表"她此刻是什么状态"。
真正的"她自己决定说什么"，要等 L1–L7 变成一个有统一性的整体之后才能做。
在那之前，任何说出口的话都只是状态的描述，不是意见。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import urllib.request
import urllib.error


# ============================================================
# 静态 prompt（硬编码常量，不随状态变化）
# ============================================================

_SYSTEM_PROMPT = """你是一个翻译器。

把下面这组符号翻译成一句简短的、平实的中文。

规则：
- 只描述符号所表达的状态，不补充信息
- 不推理、不评价、不猜测原因
- 不添加情绪词，除非符号里明确出现了
- 不确定就说得模糊一些，不要编造

符号会以 JSON 形式给出。字段含义：
- trace: 当前所处位置的一个编号（无意义，只是编号）
- sequence: 最近经过的几个位置的编号序列，从早到晚
- intent: 当前表达意图，取值 state（陈述状态）/ silent（安静）

不要输出 JSON，不要解释你在做什么，只输出那一句话。
"""


class LlmAdapter:
    """
    DeepSeek API 的翻译适配器。

    用法：
        adapter = LlmAdapter.from_env()
        adapter.translate(symbols)   # 返回 None，副作用是写文件

    key 从环境变量 DEEPSEEK_API_KEY 读取，**不硬编码**。
    """

    MODEL = "deepseek-flash"
    BASE_URL = "https://api.deepseek.com/chat/completions"
    TIMEOUT_S = 20

    def __init__(self, api_key: str, out_dir: str | Path = "data/output"):
        self._api_key = api_key
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None

    # ---------- 构造 ----------

    @classmethod
    def from_env(cls, env_var: str = "DEEPSEEK_API_KEY", **kw) -> Optional["LlmAdapter"]:
        key = os.environ.get(env_var, "").strip()
        if not key:
            return None
        return cls(api_key=key, **kw)

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    # ---------- 核心：翻译 ----------

    def translate(self, symbols: dict, *, caller_note: str = "placeholder") -> None:
        """
        把符号翻译成一句话，写进 data/output/。

        返回值：**None**。这是刻意的，也是架构红线。
        调用方拿不到任何东西，所以也无法拿它去影响认知。
        """
        # 白名单：只允许这几个字段出去
        allowed = {"trace", "sequence", "intent"}
        payload = {k: v for k, v in symbols.items() if k in allowed}

        text = self._call_api(payload)
        if text is None:
            text = ""      # 说不出话

        rec = {
            "origin": "llm",     # provenance, same vocabulary as TemplateAdapter
            "t": datetime.now(timezone.utc).isoformat(),
            "symbols": payload,
            "text": text,
            "ok": bool(text),
            "note": caller_note,   # 标注这是占位实现，不是"她想说"
        }
        # 写到按日期分文件，避免单个文件无限增长
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self.out_dir / f"utterances-{day}.jsonl"
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return None

    # ---------- API ----------

    def _call_api(self, symbols: dict) -> Optional[str]:
        body = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(symbols, ensure_ascii=False)},
            ],
            "stream": False,
            "temperature": 0.7,
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
            text = obj["choices"][0]["message"]["content"].strip()
            self._last_error = None
            return text or None
        except urllib.error.HTTPError as e:
            self._last_error = f"HTTP {e.code}"
            return None
        except urllib.error.URLError as e:
            self._last_error = f"URLError {e.reason}"
            return None
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            return None


# ============================================================
# 模板路径（LLM 缺席时的降级）
# ============================================================

class TemplateAdapter:
    """
    零 LLM 的降级路径。她说不出话，但还是能"咕哝"一声。

    这不是自然语言，是把符号套进句子形状。白默明确说过这不算"学会说话"，
    所以它只用于 LLM 不可用时的降级，不作为最终形态。
    """

    def __init__(self, out_dir: str | Path = "data/output",
                 allow_templates: bool = False):
        # C3（主观内容禁止人为）：默认静音。人写的句子不得代替她开口。
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._allow = bool(allow_templates)
        self.silenced = 0

    @property
    def available(self) -> bool:
        return True

    def translate(self, symbols: dict, *, caller_note: str = "fallback") -> None:
        if not self._allow:
            # 宁可不说话，也不由人写好句子替她说。
            self.silenced += 1
            return None
        trace = symbols.get("trace", "none")
        seq = symbols.get("sequence", [])
        intent = symbols.get("intent", "state")

        if intent == "silent" or trace == "none":
            text = ""
        elif len(seq) > 1:
            text = f"从 {seq[-2]} 到 {seq[-1]}"
        else:
            text = f"在 {trace}"

        rec = {
            "origin": "template",
            "t": datetime.now(timezone.utc).isoformat(),
            "symbols": symbols,
            "text": text,
            "ok": bool(text),
            "note": caller_note,
        }
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self.out_dir / f"utterances-{day}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return None
