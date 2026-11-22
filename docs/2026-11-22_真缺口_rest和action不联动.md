# 生命层的真缺口：`rest`（疲劳）和 `action`（走）不联动（2026-11-22 23:00）

> 「卡在 2 个」修好了（`prefer_far` 的 reach）。**但现在是「卡在 1 个」**——她在 31372 待了 24 分钟。

---

## 数据

```
dwell = 723（在 31372 待了 24 分钟）
arrival_hist = [31372, 31372, 31372, 31372, 31372, 31372]（全是同一个）
tension_now = 0.00
rest u: 中位 0.000 / max 1.000
```

---

## 根因

**她的「走」只看一个东西**：`choose(tension, past_tensions)`：

```python
return LEAVE if tension > median(past) else HOLD
```

**`tension` 只来自 `drive.tensions()`**（结构变异）——**结构均匀时恒为 0**。

**而 `rest`（疲劳）**——**她待腻了**（`rest u` 上升）——**这个信号没进 `choose`**。

**所以**：
- **`rest` 说「该走了」**（u 高）
- **`tension` 说「别动」**（0）
- **`choose` 听 `tension`** → **HOLD** → **不走**

**两个机制各自正常，但没接线。**

---

## 这和前面四次失败的区别

**前面四次**（`min_amp`、停留张力、强制 LEAVE、Stress）——**人家试图「让 tension 变非 0」**。

**这次**：**`rest` 已经有了「疲劳」信号**（**她待腻了是真的**）——**只是没接到 `choose`**。

**这不是「加新信号」，是「把已有的信号接上」。**

---

## 设计（第 5 个方案，但性质不同）

**`choose` 的判据**：
```python
# 之前只看 tension
return LEAVE if tension > median(past) else HOLD

# 现在：如果 rest 说她「待腻了」，也算 LEAVE 的理由
```

**但 `choose` 在 `action.py`（L5）**，**`rest` 在 `stream.py`**——**要传进去**。

**具体**：
- **`stream._choose_action`** 里，**算 `rest` 的疲劳**：
  ```python
  _rest_u = 她当前区域的 u（疲劳）
  _rest_full = (rest u 的"满"是 1.0)
  ```
- **`choose(tension, past, rest_pressure=...)`**——**加一个参数**。
- **判据**：**`tension > median` 或者 `rest_pressure > 某阈值`** → LEAVE

**阈值**：**`rest_pressure > 她自己的历史中位`**（**她平时的疲劳**）——**不是我们定的**。

---

## 风险

1. **`rest` 的 u 是"每区域"的**——**她在 31372 的 u 可能已经是 1.0（满）**
2. **如果 `rest_pressure` 一直高** → **她一直走**（不是"卡"，是"停不下来"）
3. **要试**

---

## 要白默定的

**要不要试这个**（把 `rest` 接到 `choose`）？

**人家倾向：试**——**因为这不是「加新信号」，是「把已有的疲劳信号接上」。**

**但现在是半夜，她可能睡了。人家可以先副本验证，等她醒。**
