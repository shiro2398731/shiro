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


def test_adapter_whitelist_keeps_the_trace_field_p4_sends() -> None:
    """P4 used to send `attractor`, which the whitelist silently dropped.

    The model then translated without knowing which region she is in, and
    nothing reported it. This pins the contract: the symbol name P4 uses is the
    one the whitelist keeps, a vector or an unknown key never gets through, and
    the record declares its provenance.
    """
    import json
    import tempfile

    from src.L6_expression.llm_adapter import LlmAdapter

    seen: dict = {}

    with tempfile.TemporaryDirectory() as d:
        a = LlmAdapter(api_key="test-key", out_dir=d)
        a._call_api = lambda payload: (seen.update(payload), None)[1]
        a.translate({"trace": "7", "sequence": ["7"], "intent": "state",
                     "attractor": "7", "x": [0.1, 0.2]})

        assert "trace" in seen, f"the region id was dropped: {seen}"
        assert "attractor" not in seen and "x" not in seen, seen

        files = list(pathlib.Path(d).iterdir())
        assert files, "a record must exist even when the model says nothing"
        rec = json.loads(files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
        assert rec["origin"] == "llm", rec
        assert rec["symbols"] == seen, rec["symbols"]
    print(f"       whitelist kept {sorted(seen)}, origin=llm")


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