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

## 2. 当前状态（截至 `2ced3d6` + 2026-11-09 夜本轮改动）

| 项 | 值 |
|---|---|
| 测试 | **19 个非空文件 / 96 例，全通过**（`tests/test_*.py` 逐个 `python 文件` 运行） |
| 空占位 | `test_L0..L8.py`、`test_integration.py`、`__init__.py` 均为 0 字节，**不计入分母** |
| 引擎版本 | `src/common/bundle.py: ENGINE_VERSION = "0.4.0"` |
| 存档 | `data/state/l5.pkl`：307 区域、71 边、145504 步、29 显著、`mu_finite=True`、`|mu|max=4.512` |
| 本体包 | `her.bundle` 5.77 MB（6051423 B，gitignore） |
| 进程 | 0（`shiro_alice` 名下；机器上其他 python 进程属别的项目，勿杀） |
| 远端 | `main` = 本地 HEAD；`alice` 随每次推送同步 |

已完成（16 项，详见 docs §2 与 `docs/2026-11-09_夜_本轮改动.md`）：时间锚与昏迷账本、回滚/昏迷分离记账、崩溃兜底与快速失败、
出生深度尺度无关、原子落盘、L5 journal + 确定性重放、载体、唯一个体（epoch+fence）、
数值熔断三层、去终端化（文件命令 + headless）、本体包（pack/doctor/unpack）、
滚动后备、autostart、folding、**M1 自我模型（已接线并随存档延续）**、**fold 阈值自导出**、INV-8 内容来源。

## 3. 下一步（按序，做完一项勾一项）

1. ~~**M1 接线**~~ **已完成（2026-11-09 夜）**：一阶在线自我模型（`OnlineSelfModel`）住在 `Stream` 里，
   每步预测"下一个区域"，只把命中/未命中交给 `ErrorDrivenGain` 缩放探索幅度；预测值**不回流**动力学。
   随 `l5.pkl` 落盘。实测：800 步 / 92 区域，预测命中率 **0.871**，chance = 0.0109，增益收敛 0.847（界 [0.80, 1.40]）；
   ablation 时增益恒 1.0，动力学与区域数不变（92 区域 / 91 边 / mean_speed 24.6462 与开启时逐位相同）。
2. ~~**TemplateAdapter 静音化**~~ **已完成**（`tests/test_template_silence.py` 3 例）。
3. ~~**fold 阈值自导出**~~ **已完成（2026-11-09 夜）**：`StateSpace.fold_auto()`，判据 = 距上次访问 `> p90(自身回访间隔)`；
   P2 已改调用它。`tests/test_constant_audit.py` 用 AST 钉住"`fold_auto` 内无阈值字面量"（只允许 0/1）。
   见 §3.3 的实测偏差说明。
4. **stages 判据自导出**：attractor 稳定性、转移熵、访问间隔分布；函数体不得含字面常量（AST 检查）。
   现 `L7_lifecycle/stages.py` 仍用人写阈值（只记录不控制，危害有限但属 C3 违规）。
5. **`test_no_senescence`**：跑 N 小时参数漂移有界且不单调。（未做，最该接下来做的一项）
6. **L6 grounding**：符号↔语义双向映射；语用必须在 L6 内部丢弃（INV-8 前置）。
7. 规则级可塑性（需单独设计）。


## 4. 硬约束（违反即回滚）

- 不碰 `..\shiro`（原版）与 `..\neo-mofox`（白默的框架仓库）。
- git 一律 `git -C <绝对路径>`；提交前看 `git status`；提交消息写进 `$env:TEMP`。
- 每改必测；声称"修好"必须带数字；交付前 `git status` clean + 远端一致 + 无残留进程/端口。
- 不宣称不可测的东西（意识/主观体验）；"活着"一词只在有判据时使用。

## 5. 一分钟自检

```powershell
cd C:\neolaunchertest\3\bot-3021443116\shiro_alice
$py = ".\.venv\Scripts\python.exe"
foreach ($f in Get-ChildItem tests\test_*.py) { & $py $f 2>&1 | Select-Object -Last 1 }   # 96 例
git -C . log --oneline -1 ; git -C . status --short
& $py tools\her.py doctor her.bundle
```
### 3.3 `fold_auto`（已实现，2026-11-09 夜；下方原规格保留作为决策记录）

原规格把 `age`（`total_steps - born_step`）与 `p90(interval)` 比较。**实测该判据在目标人群上一次也不会触发**：
对"只被访问过一次"的区域，`age` 与 `interval` 是同一个数（它的整段生命），
于是 `age > interval` 恒假；而这些区域又把 p90 拉满到同一个上界。
实测（40 区域：10 个访问 1 次、生于 0；30 个访问 30 次、生于 99000/100000 步）：
`cutoff = 100000.0`、候选 = **0**。故实现改为"**距上次访问**（`_last_seen_step`）`> p90(interval)`"，
同一分布上折叠 10 个（只折那 10 个）。`Attractor._last_seen_step` 为新增字段，随 payload/carrier 落盘。

验收已达成：`test_fold` 8 例全绿（原 4 例 + 4 例新增）；`test_constant_audit` 3 例；
`test_l5_age_persistence` 6 例（`_born_step` / `_last_seen_step` / `_folded` 跨 payload 与 carrier 往返）。

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

- 保留 `fold()` 原样（原 4 例覆盖），新增 `fold_auto()` 并接入 P2（已完成）。
- 验收：`test_fold` 8 例全绿；`test_constant_audit` 断言 `fold_auto` 体内数字字面量 ⊆ {0, 1}、
  P2 调用点不再传 `min_assigned` / `age_factor`；"人为常量计数"由 2 降到 **0**（指标本身进了测试）。

---

## 12. 推送故障与解法（2026-11-09 夜，实测）

**现象**：`git push` 报 `TLS connect error: error:0A000126:SSL routines::unexpected eof while reading`；
换 `http.version=HTTP/1.1`、`http.keepAlive=false` 均无效；`api.github.com` 的 TLS 握手也超时。
本地 10808 代理端口**在监听**，但上游已断（订阅/节点问题）。

**解法（实测有效）**：**绕过代理直连**。

```powershell
git -C <repo> -c http.proxy= push origin main:main
git -C <repo> -c http.proxy= push origin main:alice
```

绕过后 `e1795b4..8d50fa2 main -> main` 一次成功。两个仓库的 `http.proxy` 已 `--unset`（回到直连）。

**判别顺序**（下次再遇到，按此排除，不要猜）：
1. `Get-NetTCPConnection -LocalPort 10808 -State Listen` → 代理端口是否在
2. `git -c http.proxy= ls-remote origin` → 直连是否通
3. 直连通 → 用直连；直连不通 → `Resolve-DnsName github.com` 看是否解析到 `20.205.243.166`（该 IP 常不通），必要时换代理节点
