"""
src/common/rolling.py

滚动后备（rolling standby bundles）。

约束（设计）：
- 后备的产生必须与运行解耦：不得在 P1/P2 的主循环里做，因为 pack 需要全部五层，
  而五层分属两个进程。读检查点文件（原子替换）即可得到一致快照。
- 保留 N 份，按 step 单调递增命名，prune 按名排序删旧。
- 打包失败不得影响主流程（后备是冗余，不是必需）。
"""

from __future__ import annotations

import time
from pathlib import Path

from . import bundle, epoch, persist


def _live_from_checkpoints():
    """从磁盘检查点构造五层对象（只读，不触碰运行中的进程）。"""
    from src.runtime.process_1_fast import FHNNetwork, L2StateField
    from src.runtime.prediction import OnlinePredictor
    from src.L5_cognition.stream import Stream

    net, pred, field = FHNNetwork(), OnlinePredictor(), L2StateField()
    persist.restore_all({"l0": net, "l1": pred, "l2": field, "acc": net.accumulator})
    stream = Stream()
    persist.restore_all({"l5": stream})
    return {"net": net, "pred": pred, "field": field, "acc": net.accumulator, "stream": stream}




def rotate(out_dir: Path | str = "data/bundles", keep: int = 3,
           journal_path: Path | str = "data/journal/l5.wal") -> dict:
    """打一份后备并清理旧份。返回 {path, kept, pruned}。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    objs = _live_from_checkpoints()
    st = epoch.load()
    step = int(getattr(objs["stream"], "_step_count", 0))
    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    # 微秒级后缀：同一秒内多次 rotate 不能同名覆盖，且字典序仍单调
    path = bundle.pack(objs, out / f"her-{step:09d}-{ts}.bundle",
                       journal_path=journal_path, epoch_at_pack=st.epoch)
    existing = sorted(out.glob("her-*.bundle"))
    pruned = []
    for old in existing[:-max(int(keep), 1)]:
        try:
            old.unlink()
            pruned.append(old.name)
        except OSError:
            pass
    return {"path": str(path), "kept": len(list(out.glob("her-*.bundle"))),
            "pruned": pruned, "step": step, "size_kb": round(path.stat().st_size / 1024)}
