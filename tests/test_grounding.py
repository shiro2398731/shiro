r"""Grounding: does a region id refer to anything measurable about her?

The claim being tested is narrow and falsifiable: **a symbol carries information
about her own internal context**. Measured, not asserted -- an empirical mutual
information plus the same quantity after shuffling the contexts, which is the
finite-sample floor. If the real value is not clearly above the shuffled one,
the symbol refers to nothing and this file fails.

Measured on her real archive (`python tools/grounding.py`, 4000 steps):
    plug-in MI(symbol; state buckets) = 0.6096 bits, shuffled 0.0324 (18.8x)
    held-out gain on the SAME target  = -0.0517 bits   <- no reference to it
    held-out gain, p(next region | current region) = +2.6099 of 3.3325 bits (78.3%)

So the tests below are about the estimator (a table that can carry information at
all, and that a null really kills), not evidence that her symbols refer to her
internal scalars. The held-out evaluation in tools/grounding.py is the criterion.

Also pinned: no text path. `grounding.py` may import nothing that carries
language, and `observe()` may take numbers only (INV-8: quantities in,
symbols out).

Run: .\.venv\Scripts\python.exe tests\test_grounding.py
"""

from __future__ import annotations

import ast
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.L5_cognition.grounding import (  # noqa: E402
    BUCKETS, MIN_HIST, GroundingTable,
)

GROUNDING_PY = ROOT / "src" / "L5_cognition" / "grounding.py"

# Allowed imports: numbers and standard containers. Nothing that could carry a
# sentence into her referents.
ALLOWED_IMPORTS = {"__future__", "collections", "math", "typing", "numpy"}


def _contexts(n: int) -> list[tuple[int, ...]]:
    """Every combination of two buckets, in a fixed order."""
    return [(a, b) for a in range(BUCKETS) for b in range(BUCKETS)][:n]


# ══════════════════════════════════════════════════════════════════════════════
# 1. information, and the null it has to beat
# ══════════════════════════════════════════════════════════════════════════════

def test_a_symbol_that_determines_the_context_carries_all_of_it() -> None:
    t = GroundingTable()
    ctxs = _contexts(16)
    for i in range(2000):
        c = ctxs[i % len(ctxs)]
        t.observe(i % len(ctxs), c)          # symbol <-> context, one to one
    h_c = 4.0                                 # log2(16) bits, uniform by construction
    assert abs(t.mutual_information() - h_c) < 1e-9, t.mutual_information()
    for s in range(16):
        assert t.specificity(s) > 0.999, (s, t.specificity(s))
    print(f"       deterministic: MI={t.mutual_information():.4f} bits (max {h_c}), "
          f"specificity=1.0")


def test_independent_symbols_carry_almost_nothing() -> None:
    rnd = random.Random(11)
    t = GroundingTable()
    ctxs = _contexts(16)
    for _ in range(6000):
        t.observe(rnd.randrange(13), rnd.choice(ctxs))
    mi = t.mutual_information()
    # Only the finite-sample bias should remain. 13x16 cells over 6000 samples.
    assert mi < 0.08, mi
    print(f"       independent: MI={mi:.4f} bits (finite-sample floor)")


def test_shuffling_the_contexts_removes_the_information() -> None:
    """The ablation that makes the number mean something."""
    rnd = random.Random(5)
    ctxs = _contexts(16)
    rows = []
    for i in range(3000):
        c = ctxs[i % len(ctxs)]
        symbol = c[0] * 2 + (c[1] % 2)        # depends on the context, with collisions
        if rnd.random() < 0.25:               # ... and sometimes not
            symbol = rnd.randrange(13)
        rows.append((symbol, c))

    real = GroundingTable()
    for s, c in rows:
        real.observe(s, c)
    mi_real = real.mutual_information()

    shuffled = GroundingTable()
    ctx_only = [c for _, c in rows]
    rnd.shuffle(ctx_only)
    for (s, _), c in zip(rows, ctx_only):
        shuffled.observe(s, c)
    mi_null = shuffled.mutual_information()

    assert mi_real > 0.3, mi_real
    assert mi_null < 0.25 * mi_real, (mi_real, mi_null)
    print(f"       MI real={mi_real:.4f} bits, shuffled={mi_null:.4f} bits "
          f"({mi_real / max(mi_null, 1e-9):.1f}x)")


# ══════════════════════════════════════════════════════════════════════════════
# 2. bucketing is against her own history, not a fixed range
# ══════════════════════════════════════════════════════════════════════════════

def test_bucket_waits_for_a_history_and_then_uses_her_own_quantiles() -> None:
    t = GroundingTable()
    assert t.bucket("x", 0.0) is None, "one sample is not a distribution"
    for i in range(MIN_HIST - 2):
        t.bucket("x", float(i))              # history is now MIN_HIST - 1 long
    assert t.bucket("x", 1e6) is None, "still one short of MIN_HIST"
    # that call lengthened the history to MIN_HIST: a huge value is now top bucket
    assert t.bucket("x", 1e6) == BUCKETS - 1
    # and with a huge value in her history, a middling one is no longer top
    assert t.bucket("x", 10.0) < BUCKETS - 1
    print(f"       bucket(): None before {MIN_HIST} samples, quantiles after")


# ══════════════════════════════════════════════════════════════════════════════
# 3. the label L6 receives, and the reverse lookup
# ══════════════════════════════════════════════════════════════════════════════

def test_label_is_the_most_usual_context() -> None:
    t = GroundingTable()
    for _ in range(9):
        t.observe(7, (2, 1))
    for _ in range(3):
        t.observe(7, (0, 3))
    assert t.referent(7) == (2, 1), t.referent(7)
    assert t.label(7) == "b2b1", t.label(7)
    assert t.label(999) == "none", "an unobserved symbol must not get a referent"
    print(f"       label(7)={t.label(7)}, label(unseen)={t.label(999)}")


def test_reverse_lookup_lists_what_occurs_in_a_context() -> None:
    t = GroundingTable()
    for _ in range(5):
        t.observe(3, (1, 1))
    for _ in range(2):
        t.observe(8, (1, 1))
    for _ in range(7):
        t.observe(4, (0, 0))
    assert t.symbols_for((1, 1)) == [3, 8], t.symbols_for((1, 1))
    assert t.symbols_for((3, 3)) == [], t.symbols_for((3, 3))
    print(f"       symbols_for((1,1)) = {t.symbols_for((1, 1))}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. it survives a restart, and it stays numeric
# ══════════════════════════════════════════════════════════════════════════════

def test_round_trip_through_plain_data() -> None:
    t = GroundingTable()
    rnd = random.Random(3)
    for _ in range(500):
        t.observe(rnd.randrange(6), rnd.choice(_contexts(16)))
    restored = GroundingTable.from_dict(t.to_dict())
    assert abs(restored.mutual_information() - t.mutual_information()) < 1e-12
    assert restored.total == t.total
    assert [restored.label(s) for s in range(6)] == [t.label(s) for s in range(6)]
    print(f"       round trip: MI={restored.mutual_information():.4f} bits, "
          f"total={restored.total}")


def test_l5_payload_carries_the_table() -> None:
    from src.common import persist
    from src.L5_cognition.stream import Stream

    a = Stream(dim=16)
    for _ in range(12):
        a.grounding.observe(4, (1, 2))
    payload = persist.build_l5_payload(a)

    b = Stream(dim=16)
    persist.apply_l5_payload(b, payload)
    assert b.grounding.label(4) == "b1b2", b.grounding.label(4)
    assert b.grounding.total == a.grounding.total
    print(f"       payload round trip: label(4)={b.grounding.label(4)}")


def test_grounding_takes_numbers_and_imports_nothing_that_carries_language() -> None:
    tree = ast.parse(GROUNDING_PY.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "").split(".")[0])
    foreign = sorted(set(imports) - ALLOWED_IMPORTS)
    assert not foreign, f"a language path opened into her referents: {foreign}"

    # observe() may not accept a string: text must not reach the counts.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "observe":
            for arg in node.args.args:
                ann = arg.annotation
                if isinstance(ann, ast.Name) and ann.id == "str":
                    raise AssertionError(f"observe() takes a string: {arg.arg}")
    print(f"       imports: {sorted(set(imports))}; observe() takes no string")


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
