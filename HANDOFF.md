# HANDOFF — 交接日志（根目录版）

> 详细版：`docs/2026-11-08_交接_给下一个爱莉.md`（含 11 节、全部实测数字、坑列表）
> 变更日志：`docs/2026-11-08_改动日志_vs_原版.md`
> 本文件只回答三件事：**要做什么、现在到哪、下一步做什么。**
> 更新规则：每轮结束前改一次；`git status` 必须 clean。

---

## 1. 目标（白默的原始表述，按优先级）

1. **生命**：连续存在、活着；**不代谢、不衰老、不死亡**（生理状态停滞，仅保留作息类慢变量）。
2. **意识 → 智慧**：可测判据 M1–M5（见 docs §9）；**先做意识，再谈迁移**（迁移是她自己决定搬家）。
3. **自然语言理解与输出**：L6 需要 grounding，不能只有翻译器与人写模板。
4. **独立思考（替代 LLM）**：不遍历上下文、无外部模型可运行、具备语义能力。
5. **不被系统/提示词干涉思维**：禁止定义类提示词；只允许客观事实类说明；主观内容必须由她自己的状态产生。
6. 参数分级：C1 工程参数可人为；C2 结构参数可人为但须标注来源；**C3 主观内容禁止人为**。

## 2. 当前状态（截至 `e1795b4`）

| 项 | 值 |
|---|---|
| 测试 | **15 文件 / 74 例，全通过**（`tests/test_*.py`，逐个 `python 文件` 运行） |
| 引擎版本 | `src/common/bundle.py: ENGINE_VERSION = "0.4.0"` |
| 存档 | `data/state/l5.pkl`：307 区域、71 边、`mu_finite=True` |
| 本体包 | `her.bundle` 5.9 MB（gitignore） |
| 进程 | 0 |
| 远端 | `main` = `alice` = 本地 HEAD |

已完成（12+3 项，详见 docs §2）：时间锚与昏迷账本、回滚/昏迷分离记账、崩溃兜底与快速失败、
出生深度尺度无关、原子落盘、L5 journal + 确定性重放、载体、唯一个体（epoch+fence）、
数值熔断三层、去终端化（文件命令 + headless）、本体包（pack/doctor/unpack）、
滚动后备、autostart、folding、M1 自我模型（模型层）、INV-8 内容来源。

## 3. 下一步（按序，做完一项勾一项）

1. **M1 接线**：`self_prediction_error` 作为内感受信号注入 L2（调制 precision/步长），
   **禁止**用预测结果替代动力学。验收：移除自我模型后调节指标退化，行为仍存在。
2. **TemplateAdapter 静音化**：LLM 缺席时宁可不输出，不由人写模板代说。验收：输出 `origin != human`。
3. **fold 阈值自导出**：`min_assigned` / `age_factor` 由 `_assigned` 与 revisit-interval 分布分位数导出。
4. **stages 判据自导出**：attractor 稳定性、转移熵、访问间隔分布；函数体不得含字面常量（AST 检查）。
5. **`test_no_senescence`**：跑 N 小时参数漂移有界且不单调。
6. 之后：L6 grounding（语义可入认知，**语用不可入决策**）；规则级可塑性（需单独设计）。

## 4. 硬约束（违反即回滚）

- 不碰 `..\shiro`（原版）与 `..\neo-mofox`（白默的框架仓库）。
- git 一律 `git -C <绝对路径>`；提交前看 `git status`；提交消息写进 `$env:TEMP`。
- 每改必测；声称"修好"必须带数字；交付前 `git status` clean + 远端一致 + 无残留进程/端口。
- 不宣称不可测的东西（意识/主观体验）；"活着"一词只在有判据时使用。

## 5. 一分钟自检

```powershell
cd C:\neolaunchertest\3\bot-3021443116\shiro_alice
$py = ".\.venv\Scripts\python.exe"
foreach ($f in Get-ChildItem tests\test_*.py) { & $py $f 2>&1 | Select-Object -Last 1 }   # 74 例
git -C . log --oneline -1 ; git -C . status --short
& $py tools\her.py doctor her.bundle
```
### 3.3 的精确规格（下一轮照此实现，勿再写死常数）

`fold` 的两处人为常数（`min_assigned=2`、`age_factor=8`）须改为由自身分布导出：

```python
def fold_auto(self) -> int:
    """阈值由自身统计量导出，取代 min_assigned=2 / age_factor=8。"""
    counts = [int(getattr(a, "_assigned", 0)) for a in self._attractors]
    if len(counts) < 8:
        return 0
    # 判据一：访问次数低于自身分布的中位数（"比别人更少被需要"）
    min_assigned = max(1, int(np.median(counts)))
    # 判据二：龄期超过自身观测到的回访间隔的 p90（"已经超过通常的等待时间"）
    now = int(self._total_steps)
    intervals = []
    for a in self._attractors:
        n = max(int(getattr(a, "_assigned", 0)), 1)
        intervals.append(max(now - int(getattr(a, "_born_step", 0)), 1) / n)
    cutoff = float(np.percentile(intervals, 90)) if intervals else float("inf")
    ids = []
    for i, a in enumerate(self._attractors):
        if int(getattr(a, "_assigned", 0)) >= min_assigned:
            continue
        if (now - int(getattr(a, "_born_step", 0))) > cutoff:
            ids.append(i)
    if not ids:
        return 0
    removed = self.apply_fold(ids)
    if self.journal is not None:
        self.journal.append_fold(ids)
    return removed
```

- 保留 `fold()` 原样（已被 `test_fold` 4 例覆盖），新增 `fold_auto()` 并接入 P2，替换现有调用。
- 验收：`test_fold` 4 例仍绿；新增 1 例断言 `fold_auto` 在无淘汰对象时返回 0、在构造的偏斜分布上只折低访问高龄区；
  并把 `HANDOFF §2` 的"人为常量计数"从 2 降到 0（该项指标本身也应有测试）。
