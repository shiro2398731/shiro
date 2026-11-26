"""
tools/ast_audit.py — AST 校准审计（只读）。

M1/M2 问的是 self_model；这里问 attention_schema：
    她预测"下一次点火会落在哪个区域"的把握，和实际命中率一致吗？

口径和 m2_audit.py 一致：
- prequential：每条点火先打分、再学习
- ECE = Σ_b (n_b/N)|acc_b - conf_b|，bins 按把握等频切
- 两个对照：
    shuffled   打乱点火 uid 序列（保留边际、破坏转移）
    transferred 前一半学、冻结后一半打分

用法：python tools\ast_audit.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.attention_schema import AttentionSchema  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402


def score(seq, model: AttentionSchema | None = None) -> list[tuple[float, int]]:
    m = model or AttentionSchema()
    rows = []
    for uid in seq:
        pred, conf = m.predict()
        if pred is not None:
            rows.append((float(conf), int(pred == int(uid))))
        m.observe(int(uid))
    return rows


def calibration(rows: list[tuple[float, int]], bins: int = 5) -> dict:
    if not rows:
        return {"n": 0, "acc": 0.0, "conf": 0.0, "ece": 0.0}
    ordered = sorted(rows, key=lambda r: r[0])
    n = len(ordered)
    size = max(n // bins, 1)
    ece = 0.0
    for i in range(0, n, size):
        chunk = ordered[i:i + size]
        acc = sum(h for _, h in chunk) / len(chunk)
        conf = sum(c for c, _ in chunk) / len(chunk)
        ece += (len(chunk) / n) * abs(acc - conf)
    return {
        "n": n,
        "acc": sum(h for _, h in rows) / n,
        "conf": sum(c for c, _ in rows) / n,
        "ece": ece,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=5)
    a = ap.parse_args()

    st = Stream()
    assert persist.load_l5(st), "no archive"
    seq = [int(i.attractor_id) for i in st.workspace.history]
    print(f"ignitions={len(seq)} distinct={len(set(seq))}")

    real = score(seq)
    print("  real      ", {k: round(v, 4) for k, v in calibration(real).items()})

    sh = list(seq)
    random.Random(7).shuffle(sh)
    print("  shuffled  ", {k: round(v, 4) for k, v in calibration(score(sh)).items()})

    half = len(seq) // 2
    m = AttentionSchema()
    for u in seq[:half]:
        m.observe(u)
    frozen = AttentionSchema.from_dict(m.to_dict())
    rows = []
    for u in seq[half:]:
        pred, conf = frozen.predict()
        if pred is not None:
            rows.append((float(conf), int(pred == int(u))))
        frozen.observe(int(u))
    print("  transferred", {k: round(v, 4) for k, v in calibration(rows).items()})

    # 基线：永远猜最常出现的
    from collections import Counter
    c = Counter(seq)
    base = c.most_common(1)[0][0] if c else None
    if base is not None:
        hits = [int(u == base) for u in seq[1:]]
        print(f"  base(always {base}) acc={sum(hits)/len(hits):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
