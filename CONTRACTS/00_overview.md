# 契约：通信总览

> 本契约是其余所有契约的索引。它钉死四件事：进程间通信拓扑、topic 命名规范、端口分配、**事件的时序规则**。
> 任何层间通信，如果不在本契约的拓扑图中，视为架构违规。
>
> **2026-11-01 重写**：旧版全文建立在"全局 tick + 时间尺度带 + 同步屏障"之上，与 ARCHITECTURE.md §3（事件驱动，无全局节拍）冲突。本次重写废除 tick 相关的一切约定，保留拓扑、命名、端口分配的骨架。变更依据：`docs/design/00_foundations.md`、`ARCHITECTURE.md` §3。

## 1. 设计依据

- ARCHITECTURE.md §2（层间通信原则）
- ARCHITECTURE.md §3（时间原则：事件驱动，无全局节拍）
- ARCHITECTURE.md §6（进程架构）
- docs/design/00_foundations.md §3（无死量化原则）

## 2. 进程与层映射

| 进程 | 入口 | 包含的层 | 常驻 | 对外 socket |
|---|---|---|---|---|
| P1 | `src/runtime/process_1_fast.py` | L0 + L1 | 是 | PUB 5561 |
| P2 | `src/runtime/process_2_mind.py` | L2 + L4 + L5 | 是 | PUB 5562 |
| P3 | `src/runtime/process_3_life.py` | L3 + L7 + L8 | 是 | PUB 5563 |
| P4 | `src/runtime/process_4_voice.py` | L6 | 否（按需） | PUB 5560 |

**进程内通信**：P1 内 L0↔L1 通过共享 numpy 数组（毫秒级耦合需要零拷贝）；P2 内 L2/L4/L5 通过共享 `SymbolBundle` 和直接函数调用；P3 内 L3/L7/L8 通过直接函数调用和 SQLite 事务。**只有跨进程才走 socket。**

**P4 是按需启动的**：由 L4 表达驱动触发，`scripts/start_all.ps1` 不包含 P4。P4 退出后不自动重启。

**端口 5564（REP）已废弃**。同步屏障废除后，不存在需要 ack 的广播。该端口不再分配。

## 3. Topic 命名规范

格式：`<source>.<kind>[.<subkind>]`

- `source`：`l0` / `l1` / `l2` / `l3` / `l4` / `l5` / `l6` / `l7` / `l8`
- `kind`：`state` / `event` / `out`
- `subkind`：可选，小写短词

**`broadcast` 这个 kind 已废除**。旧版用它表示"发给所有人并被 ack"，那是同步屏障的产物。相位改为 L0 的普通事件（`l0.event.envelope`），各层自行订阅或不订阅。

**禁止**：跨进程 topic 不以层号开头；topic 中出现大写字母；topic 中出现空格。

## 4. 通信拓扑图

```
┌──────────────────────────────┐
│ P1: L0 + L1                  │
│ PUB 5561                     │
└──────────────┬───────────────┘
               │ l0.event.envelope      ← 相位来源，唯一
               │ l1.state.surprise
               │ l1.state.projection
               ▼
┌───────────────────────────────────────────────┐
│ P2: L2 + L4 + L5                              │
│ PUB 5562                                      │
└───────┬───────────────────────────────┬───────┘
        │ l2.state.field                │
        │ l4.state.drive                │
        │ l5.state.stream               │
        │ l5.event.transition           │
        ▼                               ▼
┌────────────────────────────┐  ┌────────────────────┐
│ P3: L3 + L7 + L8           │  │ P4: L6             │
│ PUB 5563                   │  │ PUB 5560           │
└──────────┬─────────────────┘  └─────────┬──────────┘
           │ l3.state.activation          │ l6.out
           │ l3.event.landscape_update    ▼
           │ l7.event.phase              外部消费者
           │ l7.event.stage
           │ l8.state.selfmodel
           │ l8.event.chain_node
           ▼
回到 P2（l3/l8 调制 L5 景观；l7.event.phase 只被记录）
回到 P1（l7.event.phase 作为 L0 慢层的调制参数，滞后一到多个包络周期生效）
```

**关键结构差异**：旧图中 P3 是星形中心（相位由它发出、等所有人 ack）。新图中 P1 是相位来源，P3 只是其中一个消费者。**P3 挂掉不再使系统失去相位管理。**

## 5. 端口分配表

| 端口 | 协议 | 进程 | 用途 | 订阅方 |
|---|---|---|---|---|
| 5560 | PUB | P4 | `l6.out` 表达输出 | 外部消费者（终端脚本、日志查看器） |
| 5561 | PUB | P1 | `l0.event.envelope`、`l1.state.*` | P2, P3, P4 |
| 5562 | PUB | P2 | `l2/l4/l5.state.*`、`l5.event.*` | P3, P4 |
| 5563 | PUB | P3 | `l3/l7/l8.*` | P1, P2 |
| ~~5564~~ | ~~REP~~ | — | ~~同步屏障确认~~ | **已废弃** |

端口范围固定，**不允许运行时修改**。如需变更，改本契约 + `config/default.toml` + 四个 runtime 入口，三处同步。

## 6. 各层订阅/发布清单

### P1（L0 + L1）

- **发布**：5561 上的 `l0.event.envelope`、`l1.state.surprise`、`l1.state.projection`
- **订阅**：5563 上的 `l7.event.phase`（作为 L0 慢层的调制参数，**不驱动**）
- **禁止订阅**：`l2.*`、`l3.*`、`l4.*`、`l5.*`、`l6.*`、`l8.*`、`l7.event.stage`
- **禁止发布**：任何 `l7.*` 主题。相位只能由 L0 发布，L7 不得代为发布

### P2（L2 + L4 + L5）

- **发布**：5562 上的 `l2.state.field`、`l4.state.drive`、`l5.state.stream`、`l5.event.transition`
- **订阅**：5561 上的 `l0.event.envelope`、`l1.state.*`；5563 上的 `l3.state.activation`、`l3.event.landscape_update`、`l8.state.selfmodel`、`l7.event.phase`（仅记录，不改认知轨迹）
- **禁止订阅**：`l6.*`、`l8.event.chain_node`

**注意 `l2.state.affect` 已更名为 `l2.state.field`**，payload 从 `AffectState` 改为 `StateVector`（见 `docs/design/00_foundations.md` §7）。

### P3（L3 + L7 + L8）

- **发布**：5563 上的 `l3.state.activation`、`l3.event.landscape_update`、`l7.event.phase`、`l7.event.stage`、`l8.state.selfmodel`、`l8.event.chain_node`
- **订阅**：5561 上的 `l0.event.envelope`（**这是 L7 的全部相位信息来源**）、`l1.state.surprise`（情节编码的惊讶峰值）；5562 上的 `l2.state.field`、`l4.state.drive`、`l5.state.stream`、`l5.event.transition`
- **禁止订阅**：`l6.*`

**`l7.event.phase` 的性质变了**：它不是"命令"，是"L7 对 L0 包络相位的**解读与命名**"。L0 的包络是物理事实，L7 只是在因果链里给它贴上"这是清醒段"的标签。任何层收到它时，**不得把它当成本层的开关**，只能当作一个可选的参考信号。

### P4（L6）

- **发布**：5560 上的 `l6.out`
- **订阅**：5562 上的 `l5.state.stream`（仅用于 gate 触发判断，不读取符号内容——符号通过 `stream.get_symbols()` 拉取）、`l2.state.field`、`l4.state.drive`；5563 上的 `l7.event.phase`（睡眠期不启动）
- **禁止订阅**：`l0.*`、`l1.*`、`l3.*`、`l8.*`、`l6.out`

## 7. 消息格式

所有 socket 消息是**两帧** ZeroMQ 消息：

1. **第一帧**：topic 字符串（UTF-8，无终止符）
2. **第二帧**：JSON 序列化的 payload

payload 的 schema 由各 `CONTRACTS/L*_L*.md` 定义。`src/common/types.py` 中的数据类提供 `.to_json()` / `.from_json()`，payload 必须通过这些方法序列化，**禁止手写 JSON**。

**所有 payload 必须携带 `t_event`**（事件真实发生的墙钟时刻，ISO 8601 微秒精度）。这不是序号，是时刻。接收方不得把它替换成"接收时刻"。

### 7.1 三帧扩展：因果标签（新增）

在必要 topic 上使用 **三帧**消息，第三帧是**因果标签**（causal tag）：一个由发布方生成的、短字符串标识的事件链 ID，用于在不同进程的日志之间建立因果关联。

理由：没有全局序列号之后，跨进程对齐只能靠时间戳，而时间戳无法表达"这条消息是因为那条消息而发的"。因果标签补上这一层。

**必带因果标签的 topic**：`l5.event.transition`、`l3.event.landscape_update`、`l8.event.chain_node`。

## 8. 事件的时序规则（替代旧 §8「时间尺度与 topic 的映射」）

### 8.1 没有发布频率

每个 topic 不是"以某个频率发布"，而是"**在该层状态变化到一定程度时发布**"。判据见 ARCHITECTURE.md §3.3：

```
staleness = |x_now - x_last_published| / scale(x_self) > 1 时发布
```

`scale(x_self)` 由该层自身历史的波动尺度实测得到，不是配置常量。

**例外（工程标度，允许固定）**：数值心跳的写入（见 §10），它属于 B 类工程标度，不属于认知节律。

### 8.2 订阅方的消费策略

| 策略 | 语义 | 适用 |
|---|---|---|
| **覆盖式** | 只保留最新一条，旧的丢弃 | 表示"当前状态"的 topic（`l*.state.*`） |
| **队列式** | 入队，满时丢最旧，记 `overflow` | 表示"发生过的事"的 topic（`l*.event.*`） |

**取消"按带的周期消费"**。订阅方不再"每个 Tn 消费一次"，而是在需要时非阻塞地取。取不到新消息时使用最后一次的值，并递增该值的 `staleness`。

### 8.3 各 topic 的分类

| Topic | 类别 | 消费策略 | 是否有因果标签 |
|---|---|---|---|
| `l0.event.envelope` | 事件（相位推进） | 队列式（容量 8） | 否 |
| `l1.state.surprise` | 状态 | 覆盖式 | 否 |
| `l1.state.projection` | 状态 | 覆盖式 | 否 |
| `l2.state.field` | 状态 | 覆盖式 | 否 |
| `l4.state.drive` | 状态 | 覆盖式 | 否 |
| `l5.state.stream` | 状态 | 覆盖式 | 否 |
| `l5.event.transition` | 事件 | 队列式（容量 32） | **是** |
| `l3.state.activation` | 状态 | 覆盖式 | 否 |
| `l3.event.landscape_update` | 事件 | 队列式（容量 8） | **是** |
| `l7.event.phase` | 事件（解读） | 队列式（容量 4） | 否 |
| `l7.event.stage` | 事件（里程碑） | 覆盖式 | 否 |
| `l8.state.selfmodel` | 状态 | 覆盖式 | 否 |
| `l8.event.chain_node` | 事件 | 队列式（容量 128） | **是** |
| `l6.out` | 输出 | 外部消费者自定 | 否 |

### 8.4 陈旧度传播

每个层在读取上游状态时，必须同时知道该值的**年龄**（`now - t_event`）。当上游长期不来消息时，该层的行为定义为：

- **照常演化**，不阻塞、不报错、不进入降级模式
- 上游值的 `staleness` 自然增大，按 ARCHITECTURE.md §3.3 的同一判据，该层会更少地因上游而重算——它的演化逐渐由自身动力学主导
- 超过自身 `scale` 的若干倍时，记一条 `upstream_stale` 日志，但**不改变行为**

这条规则保证：**上游挂掉，下游不会死，只会变得更"自己"。**

## 9. 启动顺序

`scripts/start_all.ps1` 的启动顺序：

```
P1 (L0+L1) ← 最先启动，它是唯一的相位来源
  等待 P1 的 5561 PUB 就绪（最多 5 s）
P3 (L3+L7+L8) ← 第二启动
P2 (L2+L4+L5) ← 第三启动，消费 P1 和 P3
P4 不启动，由 L4 表达驱动触发
```

**与旧版的差异**：旧版要求 P3 最先（因为它管相位）。现在 P1 最先。

**顺序的严格程度也变了**：P2 启动时若 P3 不在线，不会失败，只是暂时没有记忆调制和身份反馈，L5 照常演化。P3 启动时若 P2 不在线，同理。

**停止顺序**：P2 → P1 → P3。P3 最后停止，因为它持有 SQLite 事务和因果链的写锁。

## 10. 存活检测

**不使用定时心跳文件**。旧版"每 5 秒写一次 heartbeat 文件"是一种节拍，虽然属于工程标度，但它会诱导下游按检查周期写逻辑。

**新机制：事件驱动的心跳。**

- 每个进程维护一个"最后事件时刻" `last_activity`，**每发出或接收一条事件就更新它**，不写文件、不定时。
- 进程通过一个轻量的本地 socket（`\\.\pipe\shiro_<pid>` 或回环 socket）响应 `ping`，返回 `last_activity`。任何进程可以随时 ping 任何进程。
- 判定"挂掉"的判据：`now - last_activity` 超过该进程自身历史活动间隔的某个分位（分位由历史实测，不是常数）。
- 判定"活着但静默"与"挂掉"必须区分：**一个安静的系统不是死系统**。若 `now - last_activity` 大，但 ping 能得到响应且返回的 `last_activity` 在推进，则它是活的。

这一条直接服务于聋哑测试：**切掉输入后系统必然变安静，但绝不能因此被判定为死亡。**

## 11. 故障隔离

每个进程独立崩溃，**不拖垮其他进程**：

| 故障 | 影响 | 不发生的坏事 |
|---|---|---|
| P1 崩溃 | 相位源消失。P2 的 L5 继续按自身动力学演化，只是失去惊讶调制；P3 记录 `p1_lost`。L0 是唯一的不可替代品 | ~~全系统失去相位管理~~（不再可能，P3 不承担此责） |
| P2 崩溃 | P1 继续跑；P3 的情节编码暂停；P4 不被触发 | 无 |
| P3 崩溃 | P1/P2 完全不受影响（**这是本次重写最重要的改善**）。L3 的记忆调制与 L8 的身份反馈暂时缺失 | ~~500 ms 内收不到相位就进入自由运行模式~~（不再需要） |
| P4 崩溃 | 无影响 | 无 |

**重启策略**：`start_all.ps1` 的看门狗只重启确认挂掉的进程（按 §10 判据），不重启"安静但活着"的进程。

## 12. 验收测试

`tests/test_integration.py` 必须包含：

1. 四个进程按 §9 顺序启动后，所有 PUB socket 在 10 s 内可连接
2. 每个进程的订阅列表与 §6 完全一致（用 `bus.list_subscriptions()` 检查）
3. **任何层订阅 `l6.out` 时抛 `ArchitectureViolation`**
4. **任何层向 `l7.*` 发布时抛 `ArchitectureViolation`**（相位只能由 L0 发布）
5. **所有跨进程 payload 必须含 `t_event`**，缺失时抛 `SchemaMismatch`
6. 杀死 P3 后，P1 和 P2 在 30 s 内**不产生任何错误日志**，L5 的跳盆序列继续增长
7. 杀死 P1 后，P2 与 P3 在 30 s 内不退出，且 P3 记录 `p1_lost`
8. **聋哑测试**：断掉一切外部输入 30 分钟，L0 仍振荡、L1 权重仍在变、L5 跳盆序列非空（见 `docs/design/00_foundations.md` §5）
9. **无死量化扫描**：`src/` 下不出现 `time.sleep(`、`asyncio.sleep(`、`tick % k`，除非在 `tests/test_no_dead_quant.py` 的白名单内并注明属于 A/B 类

## 13. 版本与变更

本契约的任何端口、topic、订阅关系、时序规则变更，必须同步更新：

- 本文件
- `config/default.toml` 的 `[network]` 段
- 四个 `src/runtime/process_*.py` 的 socket 绑定和订阅列表
- `docs/design/10_timing_bus.md`
- `src/common/types.py`（若涉及 payload 结构）

不允许多版本并存。
