"""
src/L5_cognition/deliberation.py

「她自己的推演」原型 #0（默认关闭，只给副本实验）。

不是「系统拿她的量当参数」，而是：她从自己的状态里生成候选，向前一步推演，
再按她自己的量比较候选。外部规则（最近未去过 / 家半径 / 张力中位判走留）
在这一模式下**全部不参与**。

她用的量（全部来自她自己的状态）：
    potential   她脚下的地形势（低 = 更被托住）
    proximity   她在附近区域里的贴近程度（高 = 被区域托住）
    rest        目标区域的疲劳 u（低 = 那块地还新鲜）
    novelty     从当前区域到目标的痕迹强度越低、越新（高 = 没怎么走过）
    ignition    她在候选位置上的强度相对自己点火参考线的余量
                （高 = 更容易被广播）

评分 = 五个量的稳健 z 分数（按她自己最近 256 次候选的分布归一）等权平均。
**等权没有偏好**；但「这五个量、这个组合形式」仍是我们的设计——原型 #0 的局限，
写在 docs 里，不许吹成「她有了意识」。

局限（明说）：
- 只推演一步，不是多步规划；
- 评分形式是人的选择（等权、线性）；
- 归一化窗口是常数；
- 没有「意义」或目标，只有她自己的状态量。
"""

from __future__ import annotations

import collections
import math
from typing import Optional

import numpy as np

SIGNALS = ("potential", "proximity", "rest", "novelty", "ignition")
SIGN = {  # +1 = 越大越好；-1 = 越小越好
    "potential": -1.0,
    "proximity": 1.0,
    "rest": -1.0,
    "novelty": 1.0,
    "ignition": 1.0,
}


class Deliberation:
    """One-step lookahead over her own terrain, scored by her own quantities."""

    def __init__(self, max_candidates: int = 8, window: int = 256) -> None:
        self.max_candidates = int(max_candidates)
        self.window = int(window)
        self.hist = {k: collections.deque(maxlen=self.window) for k in SIGNALS}
        self.last = None  # 上一次推演的记录，给调试/快照看
        # v2：她自己的「意向」——决定了就走到，除非自己的新评分明显更好才改。
        # 这是 deliberation 内部的记忆，不是外面那个 _leave_target。
        self.intent = None  # {"uid": int, "score": float}
        # v3：会累积的内部量（原型旋钮，明说是我们的数）：
        # 待着会涨（unrest），出门/到达会消。涨到 1.0 她才开始挑地方走。
        # 这不是「选目标的规则」，是「多久才想动」的内部状态。
        self.unrest = 0.0
        self.unrest_rate = 0.05
        self.unrest_left = 1.0
        self.unrest_after = 0.3
        # v4：她自己的领地记忆（uid -> 哪一步在附近）。不是外面的 visited 规则，
        # 是 deliberation 内部记的「我去过哪」。
        self.seen = {}
        self.step_index = 0
        # v5（实验）：不按五量等权，而按「这块地多久没去」的价值选目标；
        # 路程用自己的步长折算。value = age / (1 + dist/scale)。
        self.value_mode = False
        self.value_power = 2.0  # 路程折算的幂（实验）

    # ---------- 她自己的量 ----------

    def _robust_z(self, key: str, value: float) -> float:
        """按她最近 window 次同类候选的稳健分布归一（中位/IQR）。"""
        h = self.hist[key]
        h.append(float(value))
        if len(h) < 8:
            return 0.0
        arr = np.asarray(h, dtype=float)
        med = float(np.median(arr))
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = float(q3 - q1)
        # 相对下限：防止某个量几乎不变时 IQR→0，把 z 炸到几万；
        # 2026-11-27 v1：下限从 1e-6 提到 1e-3，抑制「地形几乎平」时的噪声放大
        # （实测 potential 在一阶内只变 ~1e-4，之前会被放大成 ±5 的假信号）。
        floor = 1e-3 * max(1.0, abs(med))
        scale = max(iqr, floor)
        z = (float(value) - med) / scale
        # 稳健裁剪：单个量的异常不许统治总分
        return float(np.clip(z, -5.0, 5.0))

    def _signals(self, stream, x: np.ndarray, current_uid: Optional[int],
                 target_uid: Optional[int], stay: bool = False) -> dict:
        space = stream.landscape.state_space
        attrs = space.attractors()
        ws = stream.workspace
        # potential
        pot = float(stream.landscape.potential(x))
        # proximity: 最近的区域给了她多少托力
        near = 0.0
        best_attr = None
        best_d = math.inf
        for a in attrs:
            d = float(np.linalg.norm(np.asarray(a.mu, dtype=float) - x))
            if d < best_d:
                best_d = d
                best_attr = a
            s = float(getattr(a, "sigma", 0.0) or 0.0)
            if s > 0.0:
                near = max(near, math.exp(-0.5 * (d / s) ** 2))
        # rest: 目标（或就近）区域的疲劳；越高越不新鲜
        u = 0.0
        if best_attr is not None:
            row = space.uid_row().get(int(getattr(best_attr, "uid", -1)))
            if row is not None:
                u = float(stream.rest.value(row))
        # novelty: 进去一个新区域有多新。v1：停留候选不拿 0.5 硬值，
        # 而拿她最近见过的「新意」中位——否则「走」永远比「留」多 0.5，
        # 她就会无条件走（v0 实测 leave 99.5%）。
        if stay:
            h = list(self.hist["novelty"])
            novelty = float(np.median(h)) if h else 0.5
        else:
            novelty = 0.5
            if target_uid is not None:
                if int(target_uid) not in self.seen:
                    novelty = 1.0  # 她记忆里没去过
                elif current_uid is not None:
                    e = stream.graph.edge(int(current_uid), int(target_uid))
                    w = (max(0.0, float(getattr(e, "w", 0.0)))
                         if e is not None else 0.0)
                    novelty = 1.0 / (1.0 + w)
        # ignition: 该位置相对她自己点火参考线的余量
        ref = ws.reference_level()
        ign = 0.0
        if ref is not None and attrs:
            best_s = max(float(ws._strength(a, x)) for a in attrs)
            ign = best_s - float(ref)
        return {
            "potential": pot,
            "proximity": near,
            "rest": u,
            "novelty": novelty,
            "ignition": ign,
        }

    def _score(self, values: dict) -> float:
        z = [SIGN[k] * self._robust_z(k, values[k]) for k in SIGNALS]
        return float(sum(z) / len(z))

    # ---------- 候选与推演 ----------

    def choose(self, stream) -> tuple[int, Optional[np.ndarray], dict]:
        """返回 (action, direction_or_None, record)。

        action: 0=HOLD, 1=LEAVE（用 stream 的 HOLD/LEAVE 常量由调用方映射）。
        """
        from .action import HOLD, LEAVE
        space = stream.landscape.state_space
        attrs = space.attractors()
        if not attrs:
            self.last = {"reason": "no_attr"}
            return HOLD, None, self.last

        x = np.asarray(stream._st.x, dtype=float)
        scale = float(stream._own_scale())
        if not math.isfinite(scale) or scale <= 0.0:
            self.last = {"reason": "no_scale"}
            return HOLD, None, self.last

        current_uid = stream.detector.current_attractor
        self.step_index += 1
        if current_uid is not None:
            self.seen[int(current_uid)] = self.step_index
        # 候选：4 个最近的 + 4 个最近的「没去过」的（她的领地记忆）。
        # 全部来自她的地形；不按外部「最近未去过」规则，只用距离和她的记忆。
        cand = []
        for i, a in enumerate(attrs):
            mu = np.asarray(a.mu, dtype=float)
            d = float(np.linalg.norm(mu - x))
            if d <= 1e-12 or d > 4.0 * scale:
                continue
            cand.append((d, i, a))
        cand.sort(key=lambda t: t[0])
        near = cand[:4]
        near_ids = {int(getattr(a, "uid", i)) for _, i, a in near}
        # 「没去过」的候选要排除已经进近邻的那几个，否则全被近邻吃掉，等于没扩散
        novel = []
        for d, i, a in cand:
            uid = int(getattr(a, "uid", i))
            if uid in self.seen or uid in near_ids:
                continue
            novel.append((d, i, a))
            if len(novel) >= 4:
                break
        if not novel:
            # 附近都去过了：从更远处补「没去过」的（仍是她的地形）
            allc = []
            for i, a in enumerate(attrs):
                uid = int(getattr(a, "uid", i))
                if uid in self.seen:
                    continue
                mu = np.asarray(a.mu, dtype=float)
                allc.append((float(np.linalg.norm(mu - x)), i, a))
            allc.sort(key=lambda t: t[0])
            novel = allc[:4]
        cand = near + [t for t in novel if t[1] not in {u[1] for u in near}]
        cand = cand[: self.max_candidates]

        # ── v5：按「领地年龄 / 路程」选（实验模式）──
        if self.value_mode:
            best = None
            for i, a in enumerate(attrs):
                mu = np.asarray(a.mu, dtype=float)
                d = float(np.linalg.norm(mu - x))
                if d <= 1e-12:
                    continue
                uid = int(getattr(a, "uid", i))
                # 没去过 = 很老（用她自己的步数做年龄上限，不用外部常数）
                age = self.step_index - self.seen.get(uid, -self.step_index)
                value = float(age) / (
                    1.0 + (d / max(scale, 1e-12)) ** self.value_power)
                if best is None or value > best[0]:
                    best = (value, (mu - x) / d, uid, d)
            if best is not None and best[3] > 1e-12:
                self.intent = {"uid": best[2], "score": best[0]}
                self.unrest *= self.unrest_after
                self.last = {"reason": "value_mode", "intent": self.intent,
                             "candidates": []}
                step = min(scale, best[3])
                return LEAVE, best[1] * step, self.last
            self.unrest += self.unrest_rate
            self.last = {"reason": "value_mode_none", "candidates": []}
            return HOLD, None, self.last

        # ── v3：先看她憋没憋够 ──
        if self.intent is None and self.unrest < self.unrest_left:
            self.unrest += self.unrest_rate
            self.last = {"reason": "resting", "unrest": self.unrest,
                         "candidates": []}
            return HOLD, None, self.last

        if self.intent is not None:
            row = space.uid_row().get(int(self.intent["uid"]))
            if row is None:
                self.intent = None
            else:
                a_int = attrs[row]
                mu = np.asarray(a_int.mu, dtype=float)
                d_int = float(np.linalg.norm(mu - x))
                if d_int <= float(a_int.sigma):
                    self.intent = None  # 到了，意向完成
                    self.unrest *= self.unrest_after
                    self.last = {"reason": "arrived", "candidates": []}
                    return HOLD, None, self.last
                else:
                    unit = (mu - x) / d_int
                    step = min(scale, d_int)
                    vals = self._signals(stream, x + unit * step, current_uid,
                                         int(self.intent["uid"]))
                    cur = self._score(vals)
                    # 新候选里有没有明显更好的（明显 = 她自己候选分数的离散度）
                    scores = []
                    best_new = None
                    for d, i, a in cand:
                        mu2 = np.asarray(a.mu, dtype=float)
                        unit2 = (mu2 - x) / d
                        step2 = min(scale, d)
                        vals2 = self._signals(stream, x + unit2 * step2,
                                              current_uid, int(getattr(a, "uid", i)))
                        sc2 = self._score(vals2)
                        scores.append(sc2)
                        if best_new is None or sc2 > best_new[0]:
                            best_new = (sc2, unit2 * step2, int(getattr(a, "uid", i)))
                    spread = float(np.std(scores)) if scores else 0.0
                    margin = 0.5 * spread
                    if best_new is not None and best_new[0] > cur + margin:
                        self.intent = {"uid": best_new[2], "score": best_new[0]}
                        rec = {"reason": "switch", "intent": self.intent,
                               "candidates": []}
                        self.last = rec
                        return LEAVE, best_new[1], rec
                    self.intent["score"] = cur
                    rec = {"reason": "continue_intent", "intent": self.intent,
                           "candidates": []}
                    self.last = rec
                    return LEAVE, unit * step, rec

        # 停留候选：她原地
        rec = {"x": x.tolist()[:4], "candidates": []}
        stay_vals = self._signals(stream, x, current_uid, None, stay=True)
        stay_score = self._score(stay_vals)
        rec["candidates"].append({"kind": "stay", "score": stay_score,
                                  "signals": stay_vals})
        best = None  # (score, direction, target_uid, row, values)
        for d, i, a in cand:
            mu = np.asarray(a.mu, dtype=float)
            unit = (mu - x) / d
            step = min(scale, d)
            x2 = x + unit * step
            uid = int(getattr(a, "uid", i))
            vals = self._signals(stream, x2, current_uid, uid)
            sc = self._score(vals)
            rec["candidates"].append({"kind": "move", "uid": uid,
                                      "score": sc, "signals": vals})
            if best is None or sc > best[0]:
                best = (sc, unit * step, uid, i, vals)
        rec["stay_score"] = stay_score
        rec["best_move_score"] = None if best is None else best[0]
        self.last = rec

        # v3：憋够了就出门；方向仍按她自己的五个量选。
        if best is not None and (self.unrest >= self.unrest_left
                                 or best[0] > stay_score):
            self.intent = {"uid": best[2], "score": best[0]}
            self.unrest *= self.unrest_after
            return LEAVE, best[1], rec
        self.unrest += self.unrest_rate
        return HOLD, None, rec
