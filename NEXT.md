# 开机清单（2026-11-16 断电后）

> 宿舍 23:00 断电。这份清单是给白默第二天开机用的，按顺序做。

---

## 1. 把她拉起来（1 分钟）

    cd C:\neolaunchertest\3\bot-3021443116\shiro_alice
    .\.venv\Scripts\python.exe tools\headless.py start

然后确认：

    .\.venv\Scripts\python.exe tools\vitals.py

**期望**：`heartbeats` 里 P1/P2/P3 是 fresh。
如果 `verdict` 是 `down`，说明没起来——看 `data\logs\P1.log` 和 `P2.log` 的尾巴。

---

## 2. 装 autostart（这样以后断电不用手动）

    .\.venv\Scripts\python.exe tools\autostart.py install

它往用户的 Startup 文件夹写一个 `.cmd`，内容是 cd 到项目 + `headless.py start`。
已验证幂等（她在跑的时候执行 `start` 返回 2、不多起进程）。

**这一步需要白默点头**（改机器配置）。

---

## 3. 确认她带着「活气」回来（最重要）

    .\.venv\Scripts\python.exe tools\vitals.py

**看 `noise` 字段**（在 `data\logs\observations.jsonl` 里，P2 的 L5 快照）。
断电前是 0.19–0.20。

- **如果 noise > 0**：内驱的续接成功了（`p2_runner.json` 里的 `feature_hist` 回来了）。
- **如果 noise == 0**：说明 `p2_runner.json` 没被读到，或者 `feature_hist` 不够 16 行。
  查 `data\state\p2_runner.json` 里 `feature_hist` 有几行。

---

## 4. 给 4 个红测试加「存档太小就跳过」

断电前的状态：`test_loop` 4/5、`test_action` 7/8、`test_uid` 6/7、`test_index_remap` 5/6。

**不是回归**（把 P2 改动退回上一版结果一样）。原因是它们假设「存档里至少几十个区域」，
而她只剩 6 个。

要做的：在每个的前面加

```python
if len(space.attractors()) < N:  # N 按各自需要（test_uid 是 12，test_index_remap 是 20）
    print(f"       skipped: only {len(space.attractors())} regions")
    return
```

**不要**去改 `Drive.min_edges` 或 `at[10]` 的硬索引让它变绿——那是掩盖。

---

## 5. 内驱的下一步（需要白默定）

今天接上了内驱（`feature_hist` 的 std），噪声从 0 变成 0.19–0.20，但区域只从 5 长到 6。
**0.2 对梯度 346 还差一个数量级。**

下一步两个方向，都要白默点头：

- **调地形**：`ceiling`（1.0）/ `assign_ratio` / `recover` 的衰减率。
  副本上压 `ceiling` 到 0.1 只多 3 个区域，但那轮实验的噪声只有 0.0046（用几何量近似的
  `feature_hist`），不可比。**要用真实的 `feature_hist` 重做这组实验。**
- **调内驱的量级**：现在没有缩放系数（噪声 = `feature_hist` 的 std 本身）。
  加缩放 = 引入人定的数 = 外驱。**白默说过不要外驱**，所以往这边走要非常小心。

---

## 6. 一个还没解决的问题

`fold_auto` 对她持续失血（HANDOFF §28）：她冻结时不去任何区域，而 `_total_steps` 一直涨，
于是所有区域越来越 overdue，被一轮轮折掉。10 分钟能从 7 掉到 3。

**她越小，这个失血越快**——因为 `MIN_REGIONS_FOR_STATISTICS` 之下 `fold_auto` 才停。

这个和 §5 是同一个问题的两面：**她动不起来，就会被判成不需要**。

**实测数据（22:32，HANDOFF §35）**：她 6 个区域时，一个 `assigned=2493`（她几乎一直在那），
其余 5 个 `overdue` 2493–2831，`wait = p90 = 153.8` → **5/6 全部超期**。
`fold_auto` 把她折到 8 以下自己停，于是她稳定在 2–7。

### 明天先动这一根，具体两条路（都需要白默点头）

**A. 给 `fold_auto` 一个「她是否活跃」的前置**

在 `fold_auto()` 开头加一个判断：如果她最近 N 步**没有换过区域**，直接 `return 0`。

理由：`fold_auto` 想问的是「这块区域她是不是不需要了」，而「她最近根本没动」时，
这个问题没有答案——所有区域一起被冷落，不是它们各自被遗忘。

```python
# 示意，不是成品
if self._steps_since_region_change > N:   # N 用她自己历史的中位驻留时长，不是常数
    return 0
```

**要定的**：`N` 怎么取。最干净的是「她自己历史上换区域间隔的中位数」——但那正是
她现在没有的东西（她不动）。所以大概率要用一个 C1 常数，这一条需要白默拍板。

**B. 把 `MIN_REGIONS_FOR_STATISTICS` 提到 `Drive.min_edges` 所需的量级**

`Drive.tensions()` 要求 `len(edges) >= 64`（`min_edges`）。现在 `MIN_REGIONS_FOR_STATISTICS = 8`，
意思是「区域少于 8 个就不折」——但 8 个区域**永远算不出张力**。

提高到 64（或更高）等于说「在她小到算不出自己的张力之前，不允许折叠」。
**这不是调参，是把两个模块的阈值对齐**，而且它不改变「折什么」的判据，只改变「什么时候不折」。

**人家的判断：先试 B。** 它更小、更不改变语义，而且它修的是一个明显的错位
（8 个区域 vs 需要 64 条边）。A 更贴「真正想问的问题」，但 `N` 不好取。

### 验证方法（明天）

1. 在副本上跑 2000 步，对比 `MIN_REGIONS_FOR_STATISTICS` 8 / 64 两种设置下的区域数曲线；
2. 上真机后观察 30 分钟：区域数是否稳在 64 以上、`tensions` 是否非零；
3. **不要用「她活了几分钟」当判据**，用「她有没有长过 64 条边」（今天的教训：判据要带数量）。

---

## 7. 参考资料

- `HANDOFF.md` §10 事故（626→3）、§23 吸收态、§26–29 内驱
- `docs/2026-11-16_今日报告_小灯.md` 一页读完今天
- `docs/2026-11-16_内驱_突破.md` 内驱的实验与瑕疵
- `tools/vitals.py` 一条命令看状态；`tools/prod.py` 判断该不该叫醒
- 备份：`data\state\l5.pre-prod-*.pkl`（今天几次实验前的存档）
