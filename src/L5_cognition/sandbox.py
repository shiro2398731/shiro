"""
src/L5_cognition/sandbox.py

对历史/当下做一次处理，而不改动她。

为什么要有这个文件
------------------
记忆不是仓库，是**对已知信息的一次处理**——而且处理方式不止一种：

    重放   replay      从日志把过去重建出来（她已经有了，逐位一致，不过程）
    预演   rehearse    用她的动力学把当下向前推 k 步（沙盒里，不写回）
    复述   recounter   对旧观测重新算一遍符号/指代（旧画面补新索引）
    反事实 counterfact   换一个动作重跑同一段（"如果我当时离开会怎样"）

这里面只有"重放"是不产生新信息的；另外三种都会**从已知里算出未知**，
这与"把已知信息提取出来过一遍"是同一件事。

两条硬约束
----------
1. **处理历史不得篡改历史**。日志只追加；任何派生记录都必须能由日志重算出来。
   否则记忆会变成自我伪造。
2. **沙盒不写回**。预演与反事实都在副本上跑，`project()` 结束时原件必须逐位不变
   （`tests/test_sandbox.py` 断言这一点）。

副本怎么来
----------
用她自己的存档格式：`build_l5_payload` -> `apply_l5_payload`。这条路径已经有测试
（年龄结构、折叠、自模型都能往返），所以副本 = 同一具身体、同一条未来。

副本不携带的东西（写在这里，免得以后误以为它全等）
------------------------------------------------
行动计数、停留记录、RNG 的内部状态都不在 payload 里。所以：
- 副本的噪声序列从种子重新开始，与她的实际未来**不同**；
- 预演误差里因此包含"噪声不可知"的部分——这是诚实的上限，不是实现的缺陷。
"""

from __future__ import annotations

from typing import List, Optional

from .dynamics import Dynamics


def clone(stream) -> object:
    """用存档格式复制一具身体。不改动原件。"""
    from src.common import persist
    from .stream import Stream

    payload = persist.build_l5_payload(stream)
    copy = Stream(dim=stream.D)
    persist.apply_l5_payload(copy, payload)
    return copy


def clone_fast(stream) -> object:
    """同一具身体的浅层复制：内存里拷一份，不走存档格式。

    用于她自己的循环里（每次预演都要复制一次，pickle 往返太贵）。
    与 `clone()` 的差别只是速度；两者都必须不共享可变状态，
    `tests/test_sandbox.py` 对两者都断言过"步进副本不会移动她"。
    """
    import copy

    return copy.deepcopy(stream)


def project(stream, steps: int, amplitude: float,
            force_action: Optional[int] = None) -> dict:
    """在副本上向前推演 `steps` 步，返回经过的区域序列与终点。

    原件不被触碰：调用方传进来的 `stream` 会被复制一次。
    """
    copy = clone(stream)
    if force_action is not None:
        copy.force_action = int(force_action)
    ids: List[int] = []
    for _ in range(int(steps)):
        copy.landscape.set_noise_amplitude(amplitude)
        copy.step()
        aid = copy.detector.current_attractor
        ids.append(-1 if aid is None else int(aid))
    return {
        "steps": int(steps),
        "regions": ids,
        "final": ids[-1] if ids else -1,
        "speed": float(Dynamics.speed(copy._st)),
    }


def advance(stream, steps: int, amplitude: float) -> List[int]:
    """真的往前走（不是在副本上），返回经过的区域序列。这是对照用。"""
    ids: List[int] = []
    for _ in range(int(steps)):
        stream.landscape.set_noise_amplitude(amplitude)
        stream.step()
        aid = stream.detector.current_attractor
        ids.append(-1 if aid is None else int(aid))
    return ids
