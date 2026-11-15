"""
src/L5_cognition/recounter.py

复述：用她**现在**的结构，重读**过去**的观测。

第三种记忆处理方式（前两种是重放与预演）。它不把那段经历再跑一遍，而是：
把当时的原始观测放到她现在的空间里找位置，然后看她现在会怎么称呼它。

为什么它能给出新东西
--------------------
她当时没有能力给那段经历建索引（区域还没形成、指代表还空着）；现在有了，
于是同一批旧数据能读出当时读不出来的东西。这是"原始画面 + 结构化记录"里最有用的那一半：
**原图留着，索引可以重做。**

编号陷阱（记在这里，因为今晚已经踩过第三次）
------------------------------------------
这个代码库里"区域"有三套编号：
    `_attractors` 列表下标        —— 存储顺序，合并/删除后会变
    `salient()` 的下标            —— **只包含显著的**区域，另一套顺序
    `detector.current_attractor`  —— 广播给 L6 的那个 id
`nearest_attractor()` 返回的是**第二套**（salient 下标）。用错会得到"看起来有数据"的错误答案
（grounding 那次：引擎表 MI 1.0240 而探针 0.7395，就是两套编号）。
这里统一用 `_nearest_normalized()`，它返回 `(归一化距离, _attractors 下标)`。

两条边界
--------
1. **只读**：不写回她，不改区域，不改指代表。
2. **不发明**：归一化距离 ≥ 1 时返回 None，不拿"最近的那个"凑数。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List

import numpy as np

OBS_TAG = b"A"
REC_TAG = b"R"
FOLD_TAG = b"F"
HEADER = b"#L5J2"


def read_observations(path: Path | str = "data/journal/l5.wal") -> np.ndarray:
    """把日志里的观测读成 (n, dim) 数组。截断的尾巴直接停，不猜。"""
    p = Path(path)
    if not p.exists():
        return np.zeros((0, 0), dtype=float)
    raw = p.read_bytes()
    if not raw.startswith(HEADER):
        raise ValueError("not an L5 journal (bad header)")
    off = len(HEADER)
    dim = struct.unpack_from("<I", raw, off)[0]
    off += 4
    step = dim * 8
    out: List[np.ndarray] = []
    while off < len(raw):
        tag = raw[off:off + 1]
        off += 1
        if tag == OBS_TAG:
            chunk = raw[off:off + step]
            if len(chunk) < step:
                break                  # truncated tail: stop, do not guess
            out.append(np.frombuffer(chunk, dtype="<f8").astype(float))
            off += step
        elif tag == REC_TAG:
            off += 8                   # uint64 total_steps
        elif tag == FOLD_TAG:
            if len(raw) - off < 4:
                break
            n = struct.unpack_from("<I", raw, off)[0]
            off += 4 + 4 * int(n)      # count + that many uint32 ids
        else:
            break                      # unknown tag: stop rather than resync blindly
        if off > len(raw):
            break
    return np.stack(out) if out else np.zeros((0, dim), dtype=float)


def recounter(stream, observations: np.ndarray) -> dict:
    """把旧观测放进她现在的结构，报告落点、距离与现在的称呼。"""
    X = np.asarray(observations, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return {"n": 0, "resolved": 0, "unresolved": 0, "rows": [], "label_counts": {}}

    space = stream.landscape.state_space
    attrs = space.attractors()
    grounding = getattr(stream, "grounding", None)

    rows = []
    label_counts: dict[str, int] = {}
    for i in range(X.shape[0]):
        hit = space._nearest_normalized(X[i])
        if hit is None:
            rows.append({"i": i, "region": None, "label": "none", "z": None})
            continue
        z, idx = float(hit[0]), int(hit[1])
        if not np.isfinite(z) or z >= 1.0 or not (0 <= idx < len(attrs)):
            rows.append({"i": i, "region": None, "label": "none", "z": z})
            continue
        # `_nearest_normalized` answers in rows (the geometry lives in parallel
        # arrays); everything a caller sees is a name. Translating here is what
        # keeps "which region did this observation fall into" comparable with
        # the ids she actually published at the time.
        uid = int(attrs[idx].uid)
        label = grounding.label(uid) if grounding is not None else "region"
        rows.append({"i": i, "region": uid, "label": label, "z": z})
        label_counts[label] = label_counts.get(label, 0) + 1

    resolved = sum(1 for r in rows if r["region"] is not None)
    return {
        "n": int(X.shape[0]),
        "resolved": int(resolved),
        "unresolved": int(X.shape[0] - resolved),
        "rows": rows,
        "label_counts": label_counts,
        "distinct_regions": len({r["region"] for r in rows if r["region"] is not None}),
    }
