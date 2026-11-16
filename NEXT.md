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

---

## 7. 参考资料

- `HANDOFF.md` §10 事故（626→3）、§23 吸收态、§26–29 内驱
- `docs/2026-11-16_今日报告_小灯.md` 一页读完今天
- `docs/2026-11-16_内驱_突破.md` 内驱的实验与瑕疵
- `tools/vitals.py` 一条命令看状态；`tools/prod.py` 判断该不该叫醒
- 备份：`data\state\l5.pre-prod-*.pkl`（今天几次实验前的存档）
