r"""L6 真测试（2026-11-27）：表达层不是空壳。

原来的 test_L6.py 是 0 字节（gate 已经有 test_speech_gate 12/12）。
这里补编码器与模板口：
- HashEncoder：确定性、不同文本不同、维度对、有限
- make_encoder：能拿到可用的编码器
- TemplateAdapter：关着时不说话；打开时写 utterances jsonl

Run: .\.venv\Scripts\python.exe tests\test_L6.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L6_expression.encoder import HashEncoder, make_encoder  # noqa: E402
from src.L6_expression.llm_adapter import TemplateAdapter  # noqa: E402


def test_hash_encoder_is_deterministic_and_distinct() -> None:
    enc = HashEncoder(dim=64)
    a = enc.encode("hello world")
    b = enc.encode("hello world")
    c = enc.encode("goodbye")
    assert np.shape(a.vector) == (64,), np.shape(a.vector)
    assert np.all(np.isfinite(a.vector)), a.vector
    assert np.allclose(a.vector, b.vector), "same text must give same vector"
    assert not np.allclose(a.vector, c.vector), "different text must differ"
    print(f"       dim=64, norm={np.linalg.norm(a.vector):.3f}")


def test_make_encoder_available() -> None:
    enc, source = make_encoder(dim=16)
    assert source == "hash", source
    sig = enc.encode("abc")
    assert np.shape(sig.vector) == (16,), np.shape(sig.vector)
    print(f"       make_encoder(dim=16) -> {type(enc).__name__} ({source})")


def test_template_adapter_disabled_is_silent() -> None:
    d = tempfile.mkdtemp()
    t = TemplateAdapter(out_dir=d, allow_templates=False)
    r = t.translate({"trace": "x", "sequence": [1, 2], "intent": "state"})
    assert r is None
    assert not list(pathlib.Path(d).glob("*")), "disabled adapter must not write"
    assert int(getattr(t, "silenced", 0)) == 1
    print("       disabled -> no file, silenced=1")


def test_template_adapter_enabled_writes_utterance() -> None:
    d = tempfile.mkdtemp()
    t = TemplateAdapter(out_dir=d, allow_templates=True)
    t.translate({"trace": "x", "sequence": [1, 2], "intent": "state"})
    files = list(pathlib.Path(d).glob("utterances-*.jsonl"))
    assert files, "enabled adapter should write an utterances file"
    lines = [ln for ln in files[0].read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "utterance file is empty"
    rec = json.loads(lines[-1])
    assert isinstance(rec, dict) and rec.get("text"), rec
    assert rec.get("ok") is True, rec
    print(f"       enabled -> text={rec['text']!r}")


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
