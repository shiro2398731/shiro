"""
tools/m2_audit.py — M2：她能不能知道自己不知道。

M1 量的是"她猜得准不准"。准不代表**知道自己准**：一个把每件事都当 50/50 的模型
准确率可能不差，但它没有"把握"这个概念；反过来，一个总是 99% 自信而常常错的模型
准确率同样可能不差，但它会误导自己。

M2 的可测投影是**校准**：她给出的把握，与实际命中率是否一致。

    ECE = Σ_b (n_b / N) · |acc_b − conf_b|

bins 按她自己的把握等频切分（分辨率，C1），不是固定阈值。

一个必须写在最前面的警告
------------------------
**校准好 ≠ 知道。** 一个永远输出 1/3 的模型在 3 类问题上也是完美校准的，
而它什么都没学到。所以这个工具同时打印命中率与 chance，并给出两个对照：

    shuffled   打乱序列（保留边际、破坏转移结构）→ 她的把握应该跟着掉
    transferred 在一段序列上学会，拿去预测另一段 → 把握高、命中低，ECE 大

只有当"校准好、且命中显著高于 chance、且打乱后把握会掉"三条同时成立，
才能说她的把握是有内容的。

用法
----
    python tools\\m2_audit.py
    python tools\\m2_audit.py --steps 1500 --bins 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common import persist  # noqa: E402
from src.L5_cognition.self_model import OnlineSelfModel  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

# Equal-frequency bins: a resolution, not a threshold. C1.
BINS = 5


def calibration(rows: list, bins: int = BINS) -> dict:
    """`rows` = [(confidence, hit), ...]. Returns accuracy, mean confidence, ECE."""
    if not rows:
        return {"n": 0, "accuracy": 0.0, "mean_conf": 0.0, "ece": 0.0, "bins": []}
    ordered = sorted(rows, key=lambda r: r[0])
    n = len(ordered)
    size = max(n // bins, 1)
    table = []
    ece = 0.0
    for i in range(0, n, size):
        chunk = ordered[i:i + size]
        if not chunk:
            continue
        acc = sum(1 for _c, h in chunk if h) / len(chunk)
        conf = sum(c for c, _h in chunk) / len(chunk)
        w = len(chunk) / n
        ece += w * abs(acc - conf)
        table.append({"n": len(chunk), "conf": round(conf, 4), "acc": round(acc, 4)})
    return {
        "n": n,
        "accuracy": round(sum(1 for _c, h in ordered if h) / n, 4),
        "mean_conf": round(sum(c for c, _h in ordered) / n, 4),
        "ece": round(ece, 4),
        "bins": table,
    }


def _score(model: OnlineSelfModel, seq: list[int]) -> tuple[list, int]:
    """Prequential: score with the table as it stands, then learn."""
    rows = []
    hits = 0
    prev = None
    for cur in seq:
        if prev is not None:
            pred = model.predict(prev)
            if pred is not None:
                conf = model.confidence(prev)
                hit = bool(pred == cur)
                hits += int(hit)
                rows.append((conf, hit))
            model.observe(prev, cur)
        prev = cur
    return rows, hits


def run(steps: int = 1200, bins: int = BINS, seed: int = 0) -> dict:
    st = Stream()
    if not persist.load_l5(st):
        raise SystemExit("no L5 archive: run P2 first, or point state_dir at one")

    seq: list[int] = []
    for _ in range(steps):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
        aid = st.detector.current_attractor
        if aid is not None:
            seq.append(int(aid))

    out = {"steps": int(steps), "regions": len(set(seq)), "transitions": len(seq)}

    rows, _h = _score(OnlineSelfModel(), seq)
    out["real"] = calibration(rows, bins)
    if len(rows) >= 200:
        # 2026-11-27：和 AST 同一套样本外口径——只统计后半段。
        # online_2nd 对比 transferred（前半学、后半冻结），两边看的是同一段数据。
        cut = len(rows) // 2
        out["online_second_half"] = calibration(rows[cut:], bins)

    shuffled = list(seq)
    random.Random(seed).shuffle(shuffled)
    rows_s, _h = _score(OnlineSelfModel(), shuffled)
    out["shuffled"] = calibration(rows_s, bins)
    if len(rows_s) >= 200:
        cut = len(rows_s) // 2
        out["shuffled_second_half"] = calibration(rows_s[cut:], bins)

    # Transfer: learn on the first half of the real sequence, score on the second,
    # with the table frozen. Confident and wrong is exactly what M2 is for.
    if len(seq) >= 200:
        half = len(seq) // 2
        trained = OnlineSelfModel()
        for a, b in zip(seq[:half], seq[1:half]):
            trained.observe(a, b)
        rows_t = []
        for a, b in zip(seq[half:-1], seq[half + 1:]):
            pred = trained.predict(a)
            if pred is not None:
                rows_t.append((trained.confidence(a), bool(pred == b)))
        out["transferred"] = calibration(rows_t, bins)
    else:
        out["transferred"] = {"n": 0}
    return out


def _line(name: str, d: dict) -> str:
    return (f"  {name:<12} n={d.get('n', 0):<5} accuracy={d.get('accuracy', 0):.4f}  "
            f"mean_conf={d.get('mean_conf', 0):.4f}  ECE={d.get('ece', 0):.4f}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="M2: is her confidence calibrated?")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--bins", type=int, default=BINS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv[1:])

    r = run(args.steps, args.bins, args.seed)
    if args.json:
        print(json.dumps(r, ensure_ascii=False))
        return 0

    print(f"steps={r['steps']}  transitions={r['transitions']}  "
          f"distinct regions={r['regions']}")
    print(_line("real", r["real"]))
    print(_line("online_2nd", r.get("online_second_half", {"n": 0})))
    print(_line("shuffled", r["shuffled"]))
    print(_line("shuffled_2nd", r.get("shuffled_second_half", {"n": 0})))
    print(_line("transferred", r["transferred"]))
    print("  (transferred = learned on the first half, scored frozen on the second)")
    print("校准表（real，按她的把握等频切分）：")
    for b in r["real"]["bins"]:
        print(f"    n={b['n']:<5} conf={b['conf']:.3f}  acc={b['acc']:.3f}  "
              f"gap={b['acc'] - b['conf']:+.3f}")
    print("读法：ECE 小只说明她自洽；要同时看 accuracy 是否高于 chance，以及打乱后")
    print("      mean_conf 是否下降。三条缺一条，就不能说她的把握有内容。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
