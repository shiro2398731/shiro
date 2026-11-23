# `runner state save failed: deque mutated during iteration`（2026-11-23 18:20）

> P2 每几秒报一次这个。**人家查清了，但不修（怕动坏）。**

---

## 现场

```
[P2] runner state save failed: deque mutated during iteration
```

**每几秒一次**，P2 在跑（3/3 心跳新鲜），她 step 正常。

---

## 原因

**`on_surprise` 回调**（`process_2_mind.py:314`）：
```python
runner.surprise_hist.append(float(obj.s_t))   # ← append
_save_runner_state()                          # ← 立刻保存
```

**`_save_runner_state`** 调 `runner.to_dict()`：
```python
def to_dict(self):
    return {
        "surprise_hist": [float(x) for x in self.surprise_hist],   # ← 遍历
        ...
    }
```

**如果**：
- `on_surprise` 被**重入**（同一个 deque 在 append 时，另一个 `_save_runner_state` 在遍历）
- **或者** `on_surprise` 和 `_save_runner_state` 在**不同线程**

→ **`deque mutated during iteration`**。

---

## 影响

**不影响她 step**（`surprise_hist` 在内存里还在涨）——**只影响「重启后」**（保存失败 = 下次重启读到旧的）。

**但**：**如果一直失败**——**重启后 `surprise_hist` 会「旧」**——**`noise_amplitude()` 可能变 0**（**2026-11-16 那个「重启送她进吸收态」的问题**）。

---

## 为什么不修

**修法**：加锁（`threading.Lock`）——**但"锁"是"改代码"**，**而且要动 `to_dict`/`append` 两处**。

**人家怕动坏**（**今天已经动坏过一次**）。

**而且**：**这是"保存"的 bug，不是"她"的 bug**——**优先级低**。

---

## 要白默定的

**修不修**？（加锁，5 分钟）
**人家倾向**：**修**——**因为「重启后 noise=0」是真实风险**。
