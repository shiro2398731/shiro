r"""C3：主观内容禁止人为——模板句不得代替她开口。

默认静音；显式 allow_templates=True 才产出，且必须带 origin=template。

Run: .\.venv\Scripts\python.exe tests\test_template_silence.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.L6_expression.llm_adapter import TemplateAdapter  # noqa: E402

SYMBOLS = {"trace": "12", "sequence": [3, 12], "intent": "state"}


def test_default_is_silent_and_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as d:
        a = TemplateAdapter(d)
        a.translate(SYMBOLS)
        assert a.silenced == 1
        assert list(pathlib.Path(d).iterdir()) == [], "silent mode must not write utterances"


def test_explicit_enable_produces_tagged_output() -> None:
    with tempfile.TemporaryDirectory() as d:
        a = TemplateAdapter(d, allow_templates=True)
        a.translate(SYMBOLS)
        files = list(pathlib.Path(d).iterdir())
        assert files, "explicitly enabled templates must write"
        body = files[0].read_text(encoding="utf-8")
        assert '"origin": "template"' in body, "output must declare its provenance"


def test_silence_counter_accumulates() -> None:
    with tempfile.TemporaryDirectory() as d:
        a = TemplateAdapter(d)
        for _ in range(3):
            a.translate(SYMBOLS)
        assert a.silenced == 3


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
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