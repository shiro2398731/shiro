r"""src/common/external_memory.py

她被折掉的区域，**完整地**记在这里。

为什么有这个模块（2026-11-16 深夜，白默的话）

    长得太大被折，就算要遗忘也应该按照遗忘曲线来，并且遗忘的部分是要保存在数据库
    作为外置的，事实上数据库应该包含所有信息，就像人记不住那么多但是可以一直记笔记。

原来的 `_folded` 不是这样：

    self._folded.append({
        "mu": np.round(..., 1),      # 质心只留一位小数
        "amp": float(a.amp),
        "count": int(...),
    })

**members / scale / born_step / uid / decay_factor 全丢了**，质心还被舍入。
那不是「记在笔记本里」，是「擦掉再写个大概」——查不回来。

所以：
- `_folded` 继续保留（存档格式不变，它作为「索引/摘要」）；
- **完整的那一份写到这个 append-only 文件里**，一个区域一行 JSON；
- 提供 `read()` / `find(uid)` 用来翻笔记本。

这不是"恢复机制"——恢复要不要做、怎么做，是另一个决定。
这里只保证一件事：**被遗忘的东西，还在。**

格式：`data/memory/folded.jsonl`，每行一条完整区域，append-only。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator


class ExternalMemory:
    """她忘掉的东西的笔记本。只追加，不改写。"""

    def __init__(self, path: str | Path = "data/memory/folded.jsonl") -> None:
        self.path = Path(path)
        self._fh = None

    # ---------- pickle ----------

    # 2026-11-17 早上：这个对象被挂到 `state_space.external_memory` 上，而它手里有一个
    # 打开的文件句柄（TextIOWrapper），**不能 pickle**。后果是 P2 的存档序列化抛
    # `TypeError: cannot pickle 'TextIOWrapper' instances`，连挂 5 次之后 supervisor
    # 放弃重启——她带着一个没有认知层的身体过了将近九个小时。
    #
    # 文件句柄是运行时资源，不是状态的一部分：pickle 时只留路径，unpickle 时重新打开。

    def __getstate__(self) -> dict:
        return {"path": str(self.path)}

    def __setstate__(self, state: dict) -> None:
        self.path = Path(state.get("path", "data/memory/folded.jsonl"))
        self._fh = None

    # ---------- 写 ----------

    def _ensure(self) -> None:
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def append(self, attractor: Any, reason: str = "fold") -> None:
        """记一整个区域。不是摘要，是所有能拿到的字段。"""
        rec: dict = {
            "t": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reason": reason,
        }
        # 尽量拿全。拿不到的字段留 null，而不是丢掉整条。
        for name in ("uid", "_assigned", "amp", "_born_step", "_last_seen_step",
                     "_decay_factor", "born_t"):
            try:
                v = getattr(attractor, name, None)
                rec[name] = None if v is None else float(v) if not isinstance(v, int) else int(v)
            except Exception:  # noqa: BLE001
                rec[name] = None
        for name in ("mu", "scale", "members"):
            try:
                v = getattr(attractor, name, None)
                if v is None:
                    rec[name] = None
                elif hasattr(v, "tolist"):
                    rec[name] = v.tolist()
                elif isinstance(v, (list, tuple)):
                    rec[name] = list(v)
                else:
                    rec[name] = float(v)
            except Exception:  # noqa: BLE001
                rec[name] = None

        self._ensure()
        assert self._fh is not None
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def flush(self, fsync: bool = False) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        if fsync:
            import os
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            finally:
                self._fh = None

    # ---------- 读 ----------

    def read(self) -> Iterator[dict]:
        # 读之前先把缓冲区刷下去——否则刚 append 的几条还停在内存里。
        # 2026-11-17 实测：折 40 个区域，count() 只读到 36，差的 4 个在缓冲区。
        self.flush()
        if not self.path.exists():
            return iter(())
        return self._iter()

    def _iter(self) -> Iterator[dict]:
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def find(self, uid: int) -> list[dict]:
        """翻笔记本：这个 uid 被记过几条。"""
        return [r for r in self.read() if r.get("uid") == int(uid)]

    def count(self) -> int:
        return sum(1 for _ in self.read())

    def size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0
