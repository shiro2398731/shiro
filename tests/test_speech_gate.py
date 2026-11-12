r"""The speech gate: whether she speaks at all, and nothing about what she says.

`gate.py` makes three promises in its own docstring. Each one is a test here,
because a promise that is not checkable is just a comment:

1. **No text.** If the gate could produce a sentence, "the gate" would be the
   author speaking through her (C3, docs §11). Checked on the syntax tree: every
   string constant outside a docstring must be a short identifier.
2. **No pragmatics.** The gate may not read text, an inbox, or "someone asked".
   Checked by the import list: nothing from this project, and no relative
   import, so there is no path by which content reaches its decision.
3. **No chosen rate.** The criterion is "this transition has not been described
   yet", measured against her own speech history; the only constant is a buffer
   size. The behaviour tests pin that, including what happens when the buffer
   fills.

Run: .\.venv\Scripts\python.exe tests\test_speech_gate.py
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L6_expression.gate import SPOKEN_MEMORY, SpeechGate  # noqa: E402

GATE_PY = ROOT / "src" / "L6_expression" / "gate.py"

# Longest string that can still be an identifier rather than a sentence.
IDENT_MAX = 24


def _tree() -> ast.Module:
    return ast.parse(GATE_PY.read_text(encoding="utf-8"))


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Ids of string constants that are docstrings (module, class, function)."""
    out: set[int] = set()

    def mark(node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            out.add(id(first.value))

    mark(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mark(node)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. it cannot speak, only decide
# ══════════════════════════════════════════════════════════════════════════════

def test_gate_contains_no_sentence_shaped_constant() -> None:
    tree = _tree()
    docstrings = _docstring_nodes(tree)
    sentences: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            text = node.value
            if " " in text.strip() or len(text) > IDENT_MAX:
                sentences.append(text)
    assert not sentences, f"the gate could put words in her mouth: {sentences}"
    print(f"       string constants outside docstrings: identifiers only")


def test_gate_imports_nothing_from_the_project() -> None:
    tree = _tree()
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                imports.append("." * node.level + (node.module or ""))
            else:
                imports.append(node.module or "")
    foreign = [m for m in imports if m.split(".")[0] in {"src", "tools"} or m.startswith(".")]
    assert not foreign, f"pragmatics could reach the gate through {foreign}"
    print(f"       imports: {sorted(imports)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. what it decides
# ══════════════════════════════════════════════════════════════════════════════

def test_first_transition_is_not_a_transition() -> None:
    g = SpeechGate()
    g.note_transition(3)
    assert g.should_speak() is False
    assert g.last_reason == "no_transition"


def test_a_new_transition_is_described_once() -> None:
    g = SpeechGate()
    g.note_transition(3)
    g.note_transition(5)
    assert g.should_speak() is True
    assert g.last_reason == "new_transition"
    # the same pair again: already said
    g.note_transition(3)
    g.note_transition(5)
    assert g.should_speak() is False
    assert g.last_reason == "already_said"
    print(f"       (3->5) said once, then suppressed as already_said")


def test_staying_in_one_region_says_nothing() -> None:
    g = SpeechGate()
    g.note_transition(5)
    g.note_transition(5)
    assert g.should_speak() is False
    assert g.last_reason == "no_change"


def test_unknown_region_says_nothing() -> None:
    g = SpeechGate()
    g.note_transition(5)
    g.note_transition(-1)          # P4 sends -1 when there is no region
    assert g.should_speak() is False
    assert g.last_reason == "no_region"

    h = SpeechGate()
    h.note_transition(5)
    h.note_transition("not a number")
    assert h.should_speak() is False
    assert h.last_reason == "no_transition"


def test_disabled_gate_is_silent() -> None:
    g = SpeechGate(enabled=False)
    g.note_transition(1)
    g.note_transition(2)
    assert g.should_speak() is False
    assert g.last_reason == "disabled"


def test_circling_regions_are_described_once_each() -> None:
    """A cycle 1->2->1->2 has exactly two unheard transitions, then nothing new."""
    g = SpeechGate()
    said = []
    for region in [1, 2, 1, 2, 1, 2]:
        g.note_transition(region)
        said.append(g.should_speak())
    assert said == [False, True, True, False, False, False], said
    print(f"       cycle decisions: {said}")


def test_full_memory_stops_silencing_her() -> None:
    """When every transition she remembers has been said, she never goes mute.

    The buffer drops the oldest pair, so an old transition becomes reportable
    again. That is the honest failure mode: her memory is bounded, not her
    voice switched off by a counter.
    """
    g = SpeechGate(memory=2)
    for a, b in ((1, 2), (2, 3), (3, 1)):
        g.note_transition(a)
        g.note_transition(b)
        assert g.should_speak() is True, (a, b)
    assert g.said_count == 2, g.said_count          # memory bound held
    g.note_transition(1)                            # (1,2) was evicted
    g.note_transition(2)
    assert g.should_speak() is True, "forgotten pair must become reportable again"
    print(f"       memory={g.memory}: oldest pair evicted, then reportable again")


def test_counts_match_decisions() -> None:
    g = SpeechGate()
    calls = 0
    for region in [1, 2, 2, 3, 1, 3, 3]:
        g.note_transition(region)
        g.should_speak()
        calls += 1
    assert g.utterances + g.suppressed == calls, (g.utterances, g.suppressed, calls)
    print(f"       {calls} decisions = {g.utterances} spoke + {g.suppressed} suppressed")


def test_decisions_are_deterministic() -> None:
    seq = [3, 7, 7, 9, 3, 9, 9, 12, 3, 7]
    a, b = SpeechGate(), SpeechGate()
    out_a, out_b = [], []
    for region in seq:
        for g, out in ((a, out_a), (b, out_b)):
            g.note_transition(region)
            out.append(g.should_speak())
    assert out_a == out_b, (out_a, out_b)
    assert (a.utterances, a.suppressed) == (b.utterances, b.suppressed)
    print(f"       two gates, same sequence: {sum(out_a)} utterances, identical")


def test_buffer_default_is_a_size_not_a_rate() -> None:
    """The only constant is memory, and it is a buffer length (C1)."""
    assert isinstance(SPOKEN_MEMORY, int) and SPOKEN_MEMORY > 0
    g = SpeechGate()
    assert g.memory == SPOKEN_MEMORY
    print(f"       SPOKEN_MEMORY={SPOKEN_MEMORY} (buffer length)")


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
