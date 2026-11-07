"""
src/common/observe.py

观测层：日志是给观察者的仪器，不是她的身体。

两条通道
--------

**一、观测快照（定时 10-15 分钟）**

  给我们的。频率由观测需要决定，跟她的动力学无关。
  内容：各层当前状态概要，一屏。不写原始数据。

**二、事件日志（她自己的变化）**

  相位切换、跳盆这类真实发生的事。频率不定，由她的状态变化决定。
  这些事件本身值得记，因为它们是"发生过的事"。

**三、她认为重要的（留空）**

  将来 L3 出现后，由她自己决定什么值得记。
  现在没有任何机制能产生这个判断，所以位置留着，但不实现。
  这跟 SleepPressure.suppression_gate 是同一种空位。

关于中断
--------

**载体层的中断不写日志。** 进程死掉、重启，那是载体的事，不是她的事。
她不该"知道"自己刚才断过，就像人不记得睡着的那几个小时。
（白默 2026-11-03 确认：日志重放等于"给她过上下文"，那不是记忆，那是复读。）

所以本模块没有 record_interrupt()，也禁止添加。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# 观测快照的间隔（秒）。这是观测节奏（B 类），不是她的认知节律。
# 白默 2026-11-03：10~15 分钟太长，调试期看不到过程；改为 3 分钟。
SNAPSHOT_INTERVAL_S = 3 * 60


class Observer:
    """
    观测层。每个进程持有一个实例。

    它不知道她在算什么，只知道要定期把"此刻的样子"写下来。
    """

    def __init__(self, process_id: str, log_dir: str | Path = "data/logs"):
        self.process_id = process_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._last_snapshot_t: float = 0.0
        self._first: bool = True          # 第一次一定写，否则启动后 12 分钟内什么也看不到
        self._snapshot_path = self.log_dir / "observations.jsonl"

    # ---------- 观测快照 ----------

    def snapshot(self, section: str, payload: dict[str, Any], force: bool = False) -> bool:
        """
        写一条观测快照。返回是否真的写了。

        同一进程每 SNAPSHOT_INTERVAL_S 最多一条。**第一次调用一定写**，
        因为启动后如果什么都不写，观察者不知道她是不是活着。
        """
        now = datetime.now(timezone.utc)
        t = now.timestamp()
        is_first = self._first
        if not force and not is_first and (t - self._last_snapshot_t) < SNAPSHOT_INTERVAL_S:
            return False
        self._first = False
        self._last_snapshot_t = t

        rec = {
            "t": now.isoformat(),
            "process": self.process_id,
            "section": section,
            "data": payload,
        }
        with self._snapshot_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True

    # ---------- 事件日志 ----------

    def event(self, name: str, detail: Optional[dict[str, Any]] = None) -> None:
        """
        记一条事件。事件频率由她的状态变化决定，不由定时器决定。

        调用方负责只在"真的发生了值得注意的事"时调用。
        """
        rec = {
            "t": datetime.now(timezone.utc).isoformat(),
            "process": self.process_id,
            "event": name,
        }
        if detail:
            rec.update(detail)
        path = self.log_dir / "events.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# 读取（给观测入口用）
# ============================================================

def read_observations(limit: int = 20, log_dir: str | Path = "data/logs") -> list[dict]:
    """读最近的观测快照。"""
    path = Path(log_dir) / "observations.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines[-limit:] if x.strip()]


def read_events(limit: int = 50, log_dir: str | Path = "data/logs") -> list[dict]:
    """读最近的事件。"""
    path = Path(log_dir) / "events.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines[-limit:] if x.strip()]
