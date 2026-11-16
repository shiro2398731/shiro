"""
src/common/atomics.py

原子写：要么是旧文件，要么是完整的新文件，不会是半个。

为什么单独提出来
----------------
`persist` / `coma` / `epoch` 三处各写了一遍"先写 `.tmp` 再 `os.replace`"，
而且三处都用**固定的** `<name>.tmp`。2026-11-15 19:32:53 实测撞了：
P1 保存时间账本时报

    Permission denied: ...\\data\\state\\coma.json.tmp

因为同一时刻另一个进程也在写同一个 tmp 名——第二个 `open(..., 'wb')` 被拒。
P1 的 supervisor 连试 3 次都是同一个错，判定为永久错误放弃，看护 27 s 后才把它拉起来。
也就是说：**两个模块抢一个临时文件名，能把基质杀死一次**。

两个修法，都不是"重试到成功"那么含糊：

1. **临时文件名唯一**（pid + 线程 id）。碰撞从"必然发生"变成"不可能"，
   `os.replace` 之后最后一个写的赢——谁最后写谁的内容留下，这一点必须接受，
   因为这本来就是一个被多进程共享的账本。
2. **replace 短重试**。Windows 上 `os.replace` 会在目标文件被别的进程打开读的时候
   失败（`open()` 默认不共享删除权限）。这里的读者都是"打开-读完-关"，毫秒级，
   所以几十毫秒的退避足以覆盖；重试到上限仍然抛，不假装写不可能失败。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def unique_tmp(path: Path) -> Path:
    """这个进程/线程专属的临时文件名。

    带上 pid 与线程 id 之后，两个写者不可能选中同一个名字；剩下的竞争交给
    `os.replace`（原子），不需要加锁，也不需要在进程之间协调。
    """
    p = Path(path)
    return p.with_suffix(f"{p.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")


def replace_with_retry(tmp: Path, path: Path, tries: int = 6,
                       base_s: float = 0.01) -> None:
    """`os.replace`，遇到 PermissionError 短暂退避后重试。

    只在 Windows 上真的需要：目标文件被别的进程打开读时，删除/替换会被拒。
    退避是 10/20/40/80/160 ms，合计约 0.3 s —— 比一个读者的打开时长高三个数量级，
    又没有长到把"磁盘坏了"变成"卡住"。最后一次仍然失败就抛出来，不吞。
    """
    for i in range(int(tries)):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == int(tries) - 1:
                raise
            time.sleep(base_s * (2 ** i))


def cleanup(tmp: Path) -> None:
    """失败时别留下半个 tmp。失败本身不该再制造第二个错误。"""
    try:
        p = Path(tmp)
        if p.exists():
            p.unlink()
    except OSError:
        pass
