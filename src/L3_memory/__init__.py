"""
src/L3_memory/__init__.py

L3 记忆：不是存储，是结构。

对外只暴露四个名字：
  TraceGraph   —— 结构本身（哪些痕迹与哪些痕迹相连）
  Edge         —— 一条连接
  RecallEngine —— 想起来
  Consolidator —— 让结构跟上地形

为什么没有"数据库"
------------------

docs/design/04_memory.md §1.1：
    人脑记忆不是对过去的字面复制。"记忆载体"本身一直在被替换，
    而更替时间尺度与记忆存续时间对不上。不存在一个"存放记忆的地方"。

所以 L3 不建表、不存条目、不做检索。
它只是地形之上的一层连接结构。
"""

from .trace import Edge, TraceGraph
from .recall import Recall, RecallEngine
from .consolidate import Consolidator

__all__ = [
    "Edge",
    "TraceGraph",
    "Recall",
    "RecallEngine",
    "Consolidator",
]
