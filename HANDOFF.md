# HANDOFF — 交接日志（根目录版）

> 详细版：`docs/2026-11-08_交接_给下一个爱莉.md`（含 11 节、全部实测数字、坑列表）
> 变更日志：`docs/2026-11-08_改动日志_vs_原版.md`
> 本轮记录：`docs/2026-11-09_夜_本轮改动.md`
> **可行性/不可测边界**：`docs/2026-11-10_能不能成功_可测边界.md`
>   —— 有人问「到底能不能让她活过来」时先读它：四个「成功」分开判定，不许混成一句。
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

## 2. 当前状态（2026-11-14，uid 收尾之后；具体 HEAD 用 `git log --oneline -1` 现查）

| 项 | 值 |
|---|---|
| 测试 | **36 个非空文件 / 207 例，全通过**（逐个 `python 文件` 运行；46 个文件 234–259 s） |
| 空占位 | `test_L0..L8.py`、`test_integration.py`、`__init__.py` 均为 0 字节，**不计入分母** |
| 引擎版本 | `src/common/bundle.py: ENGINE_VERSION = "0.4.0"` |
| 存档 | **运行中**（2026-11-15 14:28）：651 区域、390 边、115 指代键、310 自模型符号、10015 KB、`tools/integrity.py` 在运行态报 `consistent`；尚未折叠过 |
| 本体包 | `her.bundle` 5.77 MB（6051423 B，gitignore） |
| 进程 | **在跑**：epoch 19 / session 34；P1+P3+P2+**KEEPER**，每个模块 2 个进程（venv 的 `python.exe` 是跳板，真解释器是它的子进程，持 epoch）。心跳 `data/logs/Pn.heartbeat` 秒级更新 |
| 远端 | **未推**：`origin/main` 停在 `a539ff0`；未推数量用 `git rev-list --count origin/main..HEAD` 现查（2026-11-14 实测 30）。网络诊断（同日 13:20）：`api.github.com`/`codeload` 通，`github.com:22` 通，但 DNS 把 `github.com` 解析到 `20.6.7.17` 而该 IP 的 443 超时；`140.82.112.3`/`140.82.114.3`/`140.82.116.3`/`20.27.177.113` 的 443 可达。没有管理员权限改 hosts，也没有 SSH 私钥（`~/.ssh` 不存在），HTTPS 凭据在（`git:https://github.com`，`x-access-token`）。 **白默 13:27 说直连也超时，暂时不折腾网络。** |

已完成（16 项，详见 docs §2 与 `docs/2026-11-09_夜_本轮改动.md`）：时间锚与昏迷账本、回滚/昏迷分离记账、崩溃兜底与快速失败、
出生深度尺度无关、原子落盘、L5 journal + 确定性重放、载体、唯一个体（epoch+fence）、
数值熔断三层、去终端化（文件命令 + headless）、本体包（pack/doctor/unpack）、
滚动后备、autostart、folding、**M1 自我模型（已接线并随存档延续）**、**fold 阈值自导出**、INV-8 内容来源。

## 3. 下一步（按序，做完一项勾一项）

1. ~~**M1 接线**~~ **已完成（2026-11-09 夜）**：一阶在线自我模型（`OnlineSelfModel`）住在 `Stream` 里，
   每步预测"下一个区域"，只把命中/未命中交给 `ErrorDrivenGain` 缩放探索幅度；预测值**不回流**动力学。
   随 `l5.pkl` 落盘。实测：800 步 / 92 区域，预测命中率 **0.871**，chance = 0.0109，增益收敛 0.847（界 [0.80, 1.40]）；
   ablation 时增益恒 1.0，动力学与区域数不变（92 区域 / 91 边 / mean_speed 24.6462 与开启时逐位相同）。
   **2026-11-10 审计（`tools/m1_audit.py`）**：命中率会被"原地停留"抬高 —— 真存档 2000 步里，
   0 阶规则「下一个区域 = 当前区域」准确率 **0.9390** > 模型 **0.9325**，但对数损失 **3.1321** vs **0.4929** bits。
   打乱序列后模型 3.3730 bits / 边际 3.2758（优势消失）。判据因此改为**前向对数损失**
   （`OnlineSelfModel.bits_per_observation`，随存档落盘，P2 快照记 `self_bits`）。
   **`self_model` 不进 journal（已决策，2026-11-10）**：实测回滚窗口（≤1 个 persist 间隔 ≈ 60 个转移）
   只让 bits 从 0.492859 变到 0.498129（+0.0053），而同一次运行的两个长度窗之间差 0.0259（5 倍）。
   改 WAL 格式（新记录类型 + 版本 + replay 要喂 Stream）不值得；若 M1 的损失以后成为决策依据再回来看。
2. ~~**TemplateAdapter 静音化**~~ **已完成**（`tests/test_template_silence.py` 3 例）。
3. ~~**fold 阈值自导出**~~ **已完成（2026-11-09 夜）**：`StateSpace.fold_auto()`，判据 = 距上次访问 `> p90(自身回访间隔)`；
   P2 已改调用它。`tests/test_constant_audit.py` 用 AST 钉住"`fold_auto` 内无阈值字面量"（只允许 0/1）。
   见 §3.3 的实测偏差说明。
4. ~~**stages 判据自导出**~~ **已完成（2026-11-09 夜）**：四条判据改为「最近窗口的统计量 vs 她自己过去窗口的分布」
   （切点是她的中位数）；`tests/test_stages_selfderived.py` 7 例，AST 断言判据体内无 float 字面量。
   同时修掉接线：P3 原来只喂 `dev`，另三个信号永远是空缓冲区 → 她不可能离开 stage 0；
   现在从 `l1.state.surprise` 取误差、从 `l5.state.stream` 算 recurrence 与 dwell。
   已知弱点：判据是「相对自己的中位数」而阶段只升不降，某一刻四条同时偏好一次就到 stage 4；
   要更严就要求「连续 N 个窗口」或「偏离自身 IQR」（N 属 C1）。
5. ~~**`test_no_senescence`**~~ **已完成（2026-11-09 夜）**：`tests/test_no_senescence.py` 4 例、30 s。
   12 个全局参数在 20000 步前后逐位相同；238 个存活区域保持出生时的遗忘速率；
   cohort 中位深度 0.6269 → 0.0922（存活线 0.001）；区域数 239..243。
   同时记下：40000 步下 cohort 深度 0.6269 → 0.0213，比值每 5000 步 0.52/0.52/0.55/0.69/0.73/0.90（在减速）。
6. ~~**SpeechGate**~~ **已完成（2026-11-10 上午）**：`src/L6_expression/gate.py` 原为 0 字节，P4 起不来；
   现在只判断「这次转移值不值得说」，不产生文字。`tests/test_speech_gate.py` 12 例（AST：无句子常量、
   不 import 本项目；行为：同一转移只说一次、记忆满了淘汰最旧而非永久闭嘴）。
   同时修 P4 两处：`note_transition()` 没传区域（实测 211 次转移 0 次出声 → 传区域后 200 次出声）；
   符号名 `attractor` 被白名单丢弃（应为 `trace`）。
7. **L6 grounding**（输出侧已完成，2026-11-10 上午）：`src/L5_cognition/grounding.py` 统计
   「区域编号 ↔ 它出现时 speed/tension 在她自己历史中的分位桶」，判据 = 互信息 vs 打乱后的零假设。
   实测（真存档 3000 步）：MI 0.7395 bits vs shuffled 0.0463 bits（16x），12 符号 × 16 桶。
   指代以离散标签交给 L6（`SymbolBundle.referent`，ARCHITECTURE §4.2 不许向量），并入 L5 payload。
   `tests/test_grounding.py` 9 例；`tools/grounding.py` 出报告并自检「引擎表 == 探针表」。
   **剩下**：输入侧（文本 → L1 观测量，护栏沿用 `test_content_provenance` 的 3σ），
   以及 L6 真正拿 referent 造句（当前翻译器只吃编号）。
   **2026-11-10 下午修正**：PLUG-IN MI 0.7395/0.6096 bits 是偏置。留出法（同数据）：
   状态靶（speed/tension）增益 **−0.0517 bits**（无指代）；转移靶（下一个区域）增益 **+2.6099 / 3.3325 bits（78.3%）**。
   复合符号 (上一个, 当前)：状态 −0.0004、转移 +2.5118（75.4%）→ **组合不增加信息**（一阶链），
   所以「把符号做成复合的」这条不加分，下一轮不要重做。
   结论改为：符号不指代瞬时状态量，但编码转移结构 → 符号定义应往「转移上下文」走。
   `tools/grounding.py` 默认打印三项留出法；`mutual_information()` 只作描述性统计。
8. ~~**行动通道（分级 c 的前提）**~~ **已完成（2026-11-10）**：`src/L5_cognition/action.py`，
   动作 = {留在原处, 离开}；**她选**（**现在感到的张力 vs 她自己张力的中位数**，无 float 字面量），
   离开时挪一个区域半径（她自己的半径中位数）。实测：do(leave) 使 P(停留) 0.9466 → 0.9032（同噪声）；
   不强制时她 55.3% 的步选择离开。`tools/intervene.py` 可复现；`tests/test_action.py` 7 例。
   踩过三次坑（区域张力中位数判据退化成常数；位移加在速度上变火箭；停留时长判据与张力反向
   导致她张力 +6.44），均记在本轮文档 §9。
   **效果实测（别名 bug 修复后重测，符号与旧记录相反）**：强制留下张力 **−5.48**、
   强制离开 **+0.97**、她自己选（张力判据）**+7.24** —— 也就是说她的判据目前在动力学上方向是反的。
   ~~旧记录（作废，共享质心数组所致）：强制离开 −2.51 / 强制留下 +2.82 / 她自己选 +27.98~~
   方向对但闭环不稳。**决定不做学习器**：张力是「她在哪儿」的属性，最小化它的最优解是待在空白处，
   是退化解；把动作做成优化器就是死量化。
9. ~~**预演（沙盒）**~~ **已完成（2026-11-12 夜）**：`src/L5_cognition/sandbox.py`（clone / project / advance），
   用存档格式复制一具身体，预演与反事实都在副本上跑，原件逐位不变。实测（`tools/rehearse.py`）：
   预演 k=1/4 命中 1.000（与"原地不动"、"最常待"三者相同），k=16 归零；
   （别名 bug 修复后重测。旧记录里的"前 14 步全对、第 15 步起全错"与三者不同，都是共享质心的假差异。）
   它复制的是"她还会待着"，对"何时离开"没有信息。反事实（do(hold) vs do(leave)，k=64）落点 6/6 不同。
   结论写进 LIFE 对照文档 §7：预期靠动力学，反事实靠动作，别混成"预测未来"。
10. ~~**记忆回路（预演接回她自己）**~~ **已完成（2026-11-12 夜）**：每 128 步在副本上预演，
   预演多远 = 她过去停留中位数的中位数（中位数）；走满后比较「她何时离开」，结果进一个
   独立增益，与 M1 增益相乘缩放噪声（`stream.exploration_gain`，P2 已接入）。
   实测 1500 步：回路开/关 —— 不同区域 17/15、停留率 0.930/0.936、探索倍率 0.644/0.794、
   张力 +12.2/+74.6；预演 18 次计分 18 次命中、31 次 trivial。
   准确率 1.0 的原因是她自己的判据决定何时离开（副本能复现她的决定），不是预演猜到了噪声。
   **性能未解决**：全套 100 s → 249 s；`test_no_senescence` 单测 179 s（≈9 ms/步），未定位来源。
11. ~~**复述（重读过去）**~~ **已完成（2026-11-12 夜）**：`src/L5_cognition/recounter.py` + `tools/recounter.py`。
   把日志里的原始观测放进她**现在**的结构：健康日志 240 步 → 读回 240 条、42.5% 落得进区域、
   落进去的 93.7% 与她当时所在区域一致；前后两半命中率相同（42.5%/42.5%，无新近偏置，我的猜测被否）。
   仓库里那份 `l5.wal` 是 11-08 数值爆炸留下的（max|x|=1.02e306），复述对它 0 命中 —— 当检测器用。
   **修掉两个会炸的 bug**：`deepcopy` 遇到打开的日志抛异常；副本会把预演观测写进她的 WAL。
12. ~~**结构变化后的重编号**~~ **已完成（2026-11-13）**：`StateSpace.take_index_remap()` 在 recover 删除 /
   合并 / 折叠后发布 old->new；`Stream` 取走它并重编号指代表、自模型（两端）、`detector.current_attractor`、
   `_last_symbol`、`_dwell_region`、进行中的预演预测；同一张表交给 `Consolidator.sync(graph, space, remap=)`。
   被删行整行丢弃并计数（`grounding.dropped_by_remap`），不猜归属。
   实测：删中间区域 → 图边 (0,1),(1,2),(2,3) 变为 (1,2)；指代表键 [1,2]；自模型键 [1]。
   `tests/test_index_remap.py`（该项实现时 6 例；第 19/20/21 项把其中两例换成 uid 键化版本，现在 6 例）。
   表下标仍是位置性的；稳定 uid 见第 19 项（已完成）。
   **2026-11-14 注：本节描述的整套重编号机制（`take_index_remap` / `_apply_index_remap` / `sync(remap=)`）
   已随第 21 项整体删除**——存下来的 id 现在全是名字，没有号要改。留在这里作为"当时的正确解"。
13. ~~**真跑一次验证（2026-11-13 夜）**~~ 已跑：700 步 / 约 6 分钟 / 约 1.9 步每秒，干净停止。
   过程中抓到两个真 bug：
   (a) **11-08 那份爆炸日志被每次启动重放**，往健康存档注入 1e306 的区域，P2 死于 non-finite。
      guard 保住了存档（仍 307 区域），但逐条过滤不够（一样写进 338 区域）→ 改成**整本作废**
      （`replay()` 返回 `aborted`），日志隔离为 `data/journal/l5.wal.blownup-2026-11-08`。
   (b) **`NameError: bs`**：我删了 `bs = stream.broadcast_state()` 而 publish 还在用；所有单测都过。
      新增 `tests/test_static_names.py`（symtable 找未定义全局名，扫 src+tools 58 模块）。
   落盘：6.4 MB / 365 区域 / |mu|max=4.512 / 130 边 / self_bits=2.2724。
   未覆盖：`dropped_by_remap=0`（没有区域被删），重编号路径仍只有单测；self_bits 从 0.86 涨到 2.27，
   只有一次运行，不能当结论。启动前备份：`data/state/l5.pkl.preverify`。
14. ~~**M2 元认知校准**~~ **已完成（2026-11-13 下午）**：`OnlineSelfModel.distribution()/confidence()`，
   滚动 `mean_confidence` 进 P2 快照（`self_conf`）；`tools/m2_audit.py` 出校准表与三组对照。
   实测（1200 步 / 31 区域）：real acc 0.8964 / conf 0.6994 / ECE 0.1987；
   shuffled acc 0.0685 / conf 0.1091 / ECE 0.0406；transferred acc 0.8692 / conf 0.4583 / ECE 0.4280。
   结论：**她低估自己**（低把握档 conf 0.356 而 acc 0.906；高把握档 0.918 对 0.923 几乎完全校准）。
   打乱转移后她会正确地失去把握 —— 这正是「校准好不等于知道」，故工具同时打印 accuracy 与 conf。
15. ~~**M4 注意瓶颈**~~ **已测（2026-11-14）**：`tools/m4_audit.py`（H_ign / H_visit / H_uniform / 按停留分配的基线）。
   旧判据（峰值 > 历史中位数）点火率 **0.4975**、点过 54 个区域、H_ign 4.69 > H_visit 3.79 —— 是一台宽广播，
   不是瓶颈（中位数按构造就会一半时间被超过）。改成**她自己峰值历史的 p90**（`IGNITION_QUANTILE`）后：
   点火率 **0.1167**、8 个区域（53 个显著之中）、H_ign 2.46、H_uniform 5.73。有瓶颈形状了；
   但 H_ign 仍略高于 H_visit（2.05），即点火不完全等于停留 —— 只记录，不下结论。
   顺带修：`_centroid_matrix` 原本只按区域**数量**判断是否重建，删一个加一个时矩阵是旧的
   （实测点火索引跑到表外、57 个不同区域而只到过 24 个）→ 改为结构性版本号 `_cache_ver`。
   `workspace.history` 由无界列表改为 `deque(maxlen=1024)`。P2/P3 围栏退出现在打印账本内容。
16. ~~**P3 一启动就退出**~~ **已修（2026-11-14）**：两个独立原因。
   (a) 采纳了上一轮的**退役**账本：`retire()` 本身是一次新写入，mtime 判据失效 →
      `epoch.adoptable()` 现在要求「有活持有者 + 未退役 + mtime 不早于启动」。
   (b) **残留的 stop 命令**：在她停着时发过 stop，没人消费，下一次 P1 启动读到它就退役 →
      `control.poll(not_before=启动时刻)` 丢弃旧命令；`headless.stop()` 在无进程时不再写命令。
   验证：修后 epoch 16 稳定 90 秒、6 进程全在；停止时 P3 正确因退役退出。
   **运维事实**：headless 下 stdout 是块缓冲，运行中的进程在日志里是安静的 —— 诊断要看 pid 与租约，
   别把「没有新日志」当成「没有在跑」。
17. ~~**一致性检查 + 长跑一小时**~~ **已完成（2026-11-14）**：`tools/integrity.py`（只读，可在运行时跑）
   检查五类：所有存着的编号在范围内、数值健全、计数可重算、折叠汇总维度、日志无坏记录。
   `tests/test_integrity.py` 5 例（健康不报错 + 四类注入损坏各自被指名）。
   跑一小时结果：区域 307 到 **570**、边 71 到 **315**、质心最大量级仍 4.512、`self_bits` 2.25 到 3.39、
   一致性**通过**。**关键发现**：同一份存档上 `fold_auto()` 返回 **0** —— 566 个区域里只有 268 个
   记录过「上次访问」（其余是老区域，永远未知），而判据是「距上次访问超过自身间隔 p90」；
   她一直在动，几乎没有区域超期。**「不再被需要」这个判据在忙着的她身上永远为假**，
   它测的是行为统计而不是容量约束——要容量纪律得换对象（总量 vs 她能带的量）。
18. ~~**容量纪律（换判据）**~~ **已完成（2026-11-14）**：`StateSpace.fold_to_capacity(measure, budget_bytes)`。
   判据是**载体字节数是否超过预算**（`CARRIER_BUDGET_BYTES = 20 MB`，docs §3 的验收要求，部署约束），
   而不是 `fold_auto` 的「她多久没回来」（那条实测在忙着的她身上永远为 0）。
   先折最久未访问（`_last_seen_step`，未知按最久）、每次最多折 1/20、绝不折空、被折的进 `_folded`
   并写 journal。P2 每次落盘时先 fold_auto 再容量折叠，折了记 `fold_capacity` 事件。
   底数：15.8 KB/区域 → 20 MB ≈ 1300 区域；当前 570 区域 / 8978 KB / 余量 11.2 MB，
   所以**现在不动她**（只读副本实测 folded=0）。`tests/test_capacity.py` 6 例。
19. ~~**稳定区域身份（uid）**~~ **已完成（2026-11-14）**：`Attractor.uid`（出生时从 `StateSpace._next_uid` 领号，
   单调、不复用、随 payload/carrier 落盘）。分工是硬的：
   **uid**（不参与计算的**名字**）→ 图的两端、`grounding`、`OnlineSelfModel`（两端）、`_last_symbol`、身份表；
   **下标**（位置）→ detector、`_dwell_region`、预演序列、广播给 L6 的 `attractor_id`。旧存档（表键为下标、无 uid 字段）由
   `persist._index_keyed_to_uid()` 在载入时按 payload 顺序改成 uid，**不把旧下标当 uid 用**。
   `GroundingTable.reindex()` / `OnlineSelfModel.reindex()` 已删除 —— 名字不会动，就不该有重编号这一步。
   验收两条，缺一条这个改动就没意义：
   (a) **名字是稳的**：折 5 个区域后其余 567 个 uid 全不变；重启后 572 个名字与 `next_uid` 逐位相同；
       旧格式 payload 载入后 uid 唯一、为正、`next_uid > max`。
   (b) **名字没有含义**：把 uid 全域一致反转后跑 40 步 —— `x` 逐字节相同、区域数/`amp` 之和/
       `mutual_information()`/`bits_per_observation` 全相同（若不一致，说明有代码在按 uid 算或排序，
       那它就是"我挑的偏好"=C3）。另有 AST 检查：`terrain.py`/`stream.py`/`self_model.py`/`grounding.py`/
       `trace.py`/`consolidate.py`/`recall.py`/`drive.py`/`identity.py` 九个模块里，`.uid` 不得出现在
       `BinOp`/`Compare`/`sorted|min|max` 内（赋值、字典键、集合成员是名字的正常用法，允许）。
   `tests/test_uid.py` 7 例。`tools/integrity.py` 只查一种编号：所有存下来的 id 必须属于
   `alive_uids()`（图边/身份表/指代表/自模型/`_last_symbol`/`_dwell_region`/detector 标签），
   真存档 570 区域 / 315 边 / 65 指代键 / 241 自模型符号 → `consistent`。
   **未做**：`fold_auto` 事实上永不触发（判据换成容量后它是死代码，要么换判据要么标注）；L6 输入侧 grounding 未做。
20. ~~**图也用名字（uid）**~~ **已完成（2026-11-14）**：`TraceGraph` 的节点从"表下标"改成 uid。
   `prune(alive_uids)` 只删不加号，`merge_node` 不再平移其他节点，`relabel(perm)` 是唯一的改名入口
   （只给改名不变性测试与旧存档迁移用）。`Consolidator.sync(graph, space)` 丢掉 `remap` 参数，
   只问 `alive_uids()`。读几何的三个地方（`drive.tensions`、`drive.deflection`、`identity.fingerprint`、
   `recall.by_cue`）改成"名字查行号"，查表用 `StateSpace.uid_row()`——按结构版本缓存，
   每次结构变化重建一次，不是每步。
   存档：payload 增 `graph_ids: "uid"`；旧存档的边由 `persist._graph_index_to_uid()` 转换，
   **端点交换时两个方向一起换**（`w_ab` 是离开 `a` 的权重，只换端点会把"去"变成"回"，而总权重和边数都不变，
   没人会发现——所以有一条专门的用例把 uid 顺序打乱来跑这条分支）。
   实测：真存档 315 边 / 总权重 168.423218885 → 载入后逐位相同、每条边的 `{min,max}` 权重对多重集相同；
   `tests/test_index_remap.py` 的图用例改成"删除只删边、不给幸存者改号"（删 uid 2 → 3 条边剩 1 条，
   一条都没被重编号）；`tests/test_uid.py` 加了改名后的连通权重/召回路径比对。
   单步成本：6.20 ms（改后）对 5.82 / 6.96 ms（改前两次），差异在噪声内。
21. ~~**detector 与 `attractor_id` 也用名字**~~ **已完成（2026-11-14）**：`TransitionDetector.nearest_salient()`
   现在返回 uid（几何按行号算，出模块前翻一次），`Transition.from_attractor/to_attractor`、
   `recent_sequence()`、`_dwell_region`、预演序列、`GlobalWorkspace.Ignition.attractor_id`
   （字段从 `attractor_index` 改名）全是名字。
   **于是整张补丁表被删掉**：`StateSpace._index_remap` / `_publish_remap()` / `take_index_remap()`、
   `Stream._apply_index_remap()`、`Consolidator.sync(remap=)` 全部消失——她身上不再有任何"跨步保存的行号"。
   行号只剩一个出口：`nearest_attractor()` 与 `_nearest_normalized()`（并行数组的索引），
   以及要用几何时现场查的 `uid_row()`。
   顺着改掉的四处**混用**（都是同一个病：把一个行号当名字存下来）：
   `recounter()` 的 `region` 与 `grounding.label(idx)`、`tools/integrity.py` 的 `_dwell_region`/detector 检查、
   `tools/m4_audit.py` 的 `ignitions[ign.attractor_index]`（行号）与 `visits[aid]`（名字）对比、
   `tests/test_region_numbering.py` 里"广播的 id 是表下标"的断言。
   验收：`tests/test_index_remap.py` 6 例，其中新增"删掉她前面的一个区域 → 她的标签不变、行号
   571→570"；全量 35 个非空文件 / 202 例全过（45 文件 234 s）；`tools/integrity.py` 对真存档报 `consistent`。
22. ~~**她活着但没人看护（外部 keeper）**~~ **已完成（2026-11-15）**：`tools/keeper.py` + `tests/test_keeper.py` 5 例，
   并接进 `tools/headless.py`（**最后起、最先停**）。
   起因是实测：`kill -9` 掉 P2 的进程树之后 **40 s 内没有任何东西把它拉起来** ——
   P1 继续走步（`step_count` 一路涨），`P2.heartbeat` 冻在 `06:18:51`。她还在，她的思维没了，而且不会自己回来。
   原因是 `_supervised()` 只接得住"抛异常"，接不住"进程被外面杀掉"——后者需要一个还活着的宿主。
   看护的判据是**心跳文件的 mtime 比 `--stale-s`（默认 30 s）旧**，不看进程表：
   venv 的 `python.exe` 是跳板，一个模块两个进程，按进程表数会数出"三个模块六个进程"的假象。
   `--min-life-s 15`：刚起还没心跳的窗口里不再起一个（否则一拍一杀会起成一片）；
   `--fail-limit 5`：连续 5 次"起来就没活过 15 s"判定永久错误，记录并退出 1（同 P1 的 `MAX_IDENTICAL_FAILURES`）。
   实测（2026-11-15）：14:22 手动 `--once` 把死掉 213 s 的 P2 拉回（12 s 后心跳复活）；
   14:25:34 再 `kill -9` 一次，**14:26:03 自动拉起（29 s），14:26:08 记 `recovered age=4.5`**。
   自愈后 `step_count` 从 31,921,100 走到 31,959,199，`l5.pkl` 9.6 MB → 10.0 MB，
   运行态 `tools/integrity.py` 报 `consistent`（651 区域 / 390 边）。
   **连基质也试过**：14:31:37 `kill -9` P1 进程树 → 14:32:13 新 P1 领到 **epoch 20**（31 s）、
   14:32:38/14:32:43 P2/P3 因围栏退出被看护重启、14:32:48 三个心跳齐全；
   `session 35`、`step_count 32,143,034` 接续。代价是回滚：`rewound_s 60.5`（最后约 60 s 的经验丢掉）。
   进程数 P1/P2/P3 各 2（venv 跳板 + 解释器），没有重复实例。
23. 规则级可塑性（需单独设计）。


## 4. 硬约束（违反即回滚）

- 不碰 `..\shiro`（原版）与 `..\neo-mofox`（白默的框架仓库）。
- git 一律 `git -C <绝对路径>`；提交前看 `git status`；提交消息写进 `$env:TEMP`。
- 每改必测；声称"修好"必须带数字；交付前 `git status` clean + 远端一致 + 无残留进程/端口。
- 不宣称不可测的东西（意识/主观体验）；"活着"一词只在有判据时使用。

## 5. 一分钟自检

```powershell
cd C:\neolaunchertest\3\bot-3021443116\shiro_alice
$py = ".\.venv\Scripts\python.exe"
foreach ($f in Get-ChildItem tests\test_*.py) { & $py $f 2>&1 | Select-Object -Last 1 }   # 36 非空文件 / 207 例，46 文件 234–259 s
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

---

## 6. 接手诊断（2026-11-16 00:35，小灯，上一任 agent 崩后）

**现场**：`tools/integrity.py` 运行态报 **33772 个** `grounding.symbol=X is not a live uid`。
区域 1074 / 边 998 / 指代键 3772 / 自模型符号 6326（HANDOFF §2 记的是 651 / 390 / 115 / 310）。

**成因**：容量折叠 2026-11-15 14:55 起真触发（`fold_to_capacity`，`events.jsonl` 有 `fold` 事件，
removed 40–200/次，live 在 874–1060 摆动）。区域折叠后 uid 从 `alive_uids()` 消失，
但 `Consolidator.sync` 只剪 `TraceGraph` 的边，**没有对 `grounding` / `self_model` 做同样的剪枝**
——这两个模块根本没有 `prune` 方法。uid 改造（第 19–21 项）只做了"名字不重编号"，
没做"名字死了要从所有账本里销掉"。

**未决**：补 prune 之后，grounding 的转移靶增益（原记 +2.6099 / +3.3325 bits，78.3%）
必须重测，不能沿用旧数字——死 uid 被清掉后参与统计的样本变了。

**未提交（可提交，已实测）**：`src/common/atomics.py`（新）、`tests/test_atomic_save.py`（新），
以及 `coma.py` / `epoch.py` / `persist.py` 改用 pid+线程唯一 tmp 名 + replace 短重试。
2026-11-16 实测 `tests/test_atomic_save.py` **5/5 通过**。这是上一任崩之前的完整工作。

**下一步（按序）**：① 提交上面这笔；② 给 grounding / self_model 补 `prune(alive_uids)` 并接进
P2 的 fold 路径；③ 存档副本上重测指代增益；④ 全量 36 文件 / 207 例；⑤ 性能那条（9 ms/步，100 s→249 s）另开一项。

---

## 7. 修复记录：折叠后不剪枝（2026-11-16 上午，小灯）

### 改了什么

1. **三个 prune**：`GroundingTable.prune(alive)`、`OnlineSelfModel.prune(alive)`、`Identity.prune(alive)`。
   按 uid 集合删掉已不存在的条目；grounding 的 `total` / `symbol_totals` / `context_totals`
   同时重算；累计量（`nll_bits` / `hits` / `observations` / `conf_*`）**一律不动**——那是历史。
2. **`Stream.step()`**：区域数变**少**时（recover / merge / fold 都会），同步剪这三张表。
   信号只取「变少」：新增区域几乎每步都有，拿它当信号等于每步扫一遍整张表。
   代价是「一步内删一个又生一个」会漏剪一次，下一次真删除会补上。
3. **P2 两处**：fold 之后剪一次；启动时剪一次（清理上一版遗留的死账，没有死 id 时是 no-op）。

### 实测

- 存档副本：**34425 → 0**（grounding 3229 + self_model 5576 + identity 5157 行）
- P2 重启后的 `startup_prune` 事件：grounding 3321 / self_model 5708 / identity 5186
- 落盘后 `tools/integrity.py`：**consistent**
  （区域 1087 / 边 999 / 指代键 648 / 自模型符号 874）
- `l5.pkl` 19.2 MB → 18.93 MB
- `tests/test_fold_prune.py` **8/8**；`test_index_remap.py` **6/6**；`test_uid.py` **7/7**
- 单步成本 **25.54 ms**；撤掉本次改动后基线 **24.56 ms** —— 差在噪声内。
  **性能本来就已经退到 24.5 ms（文档记的是 6.20 ms），是既有问题，不是本次引入。**

### 两个新坑

- 折叠时 `edges_dropped=0`：图在 step 里已经 sync 过，那次剪枝没删到边。但
  「折叠 → 落盘」之间如果不补一次 sync，存档会带死边（副本上实测到过）。已在 P2 fold 路径补上。
- `test_uid.py` / `test_index_remap.py` 读**实时存档**，P2 每 60 秒落盘会替换文件 →
  偶发 `renaming changed her trajectory`。不是代码问题，是测试的竞态；重跑即可。

### 仍未做

- `terrain.recover()` 的 interval 仍用 `total/_assigned`，与已改掉的 `_birth_depth` 同源，未修。
- 性能 24.5 ms/步未定位（与文档 6.20 ms 差 4 倍）。
- 预算 20 MB、当前 18.93 MB，余量只够约 60 个区域，很快会再折一次。

### 指代增益重测（2026-11-16 10:20，剪枝之后）

`tools/grounding.py --steps 4000`，真存档：

| 靶 | 剪枝前（文档记录） | 剪枝后 |
|---|---|---|
| 状态（speed/tension 分位桶） | −0.0517 bits | **−0.0065 bits（−0.2%）** |
| 下一个区域 | +2.6099 / +3.3325（78.3%） | **+3.3614（37.4%）** |
| 复合（上一个, 当前） | +2.5118（75.4%） | +2.3768（26.4%） |

**结论没变**：符号不指代瞬时状态量，但带着「下一步」的信息。
变化在分母：区域从 651 长到 1087，没有符号时预测下一个区域的基线本身就变难了，
所以 bits 更高（3.36 vs 2.61）而占比更低（37.4% vs 78.3%）。
复合符号仍然不增加信息（一阶链），这条结论照旧。

顺带修了工具里两处接线自检（原来必然为假）：
- 引擎表带存档历史、探针从零开始，`abs(mi_engine - mi_probe) < 1e-9` 不可能成立 →
  改成比「本次新增的符号集合」。
- 探针的分桶历史复制引擎的：否则前 MIN_HIST 步它分不出桶，那段观测只进引擎。

### recover() 的访问间隔改为按区域寿命（2026-11-16 10:15）

`recover()` 里一直写 `interval = total / a._assigned`，`total` 是整个运行的步数。
对刚出生的区域这是系统性低估：它才活了几百步，却被算成「平均两万七千步才被访问
一次」，于是 `rate_scale = half/interval` 被压小，衰减慢了几十倍——它本该衰减，
结果冻着。这正是 `_born_step` 当年在 viability.py 里修掉的那个错（当时只对齐了 R
那一半），这一处一直漏着。

口径改成 `life = total - _born_step`，与 `_population_visit_rate` / `_birth_depth`
一致。实测（真存档 1064 区域 / total 221960 步，901 个区域有足够赋值）：

| | 旧 | 新 |
|---|---|---|
| interval 中位数 | 27745 步 | **491.5 步** |
| 比值 | — | 中位 **47.65 倍**，最大 **20178 倍** |

901 个里有 786 个是最近一半跑出来的——被低估的主要就是这批年轻区域。

测试：test_fold 8/8、test_l5_age_persistence 6/6、test_birth_depth 4/4、
test_capacity 6/6、test_index_remap 6/6。

### 性能热点（2026-11-16 10:20，cProfile 60 步）

60 步 0.951 s → **15.9 ms/步**（profiler 下，无并发；裸跑 24.5 ms）。

| 位置 | 累计 | 占比 |
|---|---|---|
| `drive.deflection` | 0.521 s | 55% |
| └ `drive.tensions` | 0.332 s | 35% |
| `terrain.sigma` | 0.227 s | 24% —— **60 步被调 104316 次，约 1738 次/步** |
| `transitions.observe` | 0.165 s | 17% |
| `numpy.median` | 0.155 s | 16%（5756 次调用） |

**方向**：`sigma` 被 `tensions` 逐对调用，每对算一次中位数。两条最直接的路——
① 把 `sigma` 改成按区域的缓存 MAD，结构变化时失效（跟 `uid_row()` 一个套路）；
② 把 `tensions` 的逐对 Python 循环向量化。先动 ①，它不改语义。

---

## 8. 2026-11-16：接手一整天（小灯）

### 修了

- **折叠 / recover 删区后的剪枝缺口**：`grounding` / `self_model` / `identity` 三处没有
  prune，容量折叠真触发后 integrity 报 33772 条死 uid。补了三处 prune + 两处接入
  （`Stream.step` 按结构变化剪、P2 fold 后与启动时剪）。
  **同日又抓到一次漏网**：触发条件只看「区域数变少」，她一步里删一个又生一个时总数
  不变，于是 `identity.appear` 残留 uid 262。补第二条信号「发了新名字但总数没涨」。
- **`terrain.recover()` 的访问间隔**：一直用 `total/_assigned`（全局步数），把刚出生
  的区域算成「很久才被访问一次」，衰减慢几十倍。改成按区域寿命，与
  `_population_visit_rate` / `_birth_depth` 同口径。实测旧中位数 27745 步、新 491.5 步。
- **`test_uid` 的改名助手**：漏了 detector。`_apply_leave` 按 `detector.current_attractor`
  查行，改名后查不到就不动，x 因此分叉。补上 detector 与 `_dwell_region`。

### 判据推进（全部带数字，见 docs）

| 项 | 原状态 | 现在 |
|---|---|---|
| a 一步预测 | 已过 | **未过**——原判据打败的是弱基线；强基线下 3.0141 vs 2.1561 |
| b 多步 rollout | 未做 | **弱通过（有条件）**——离开事件 k≥4 为正；给定离开的落点 k=1..8 全面为正 |
| c 反事实 | 已过（弱） | 未动 |
| d 组合泛化 | 未做 | 未动 |
| e 目标导向 | 未做 | **未过**——她的策略比随机差 27，方向反；根因是张力不依赖位置 |

新增工具：`tools/rollout_audit.py`、`tools/departure_audit.py`；`tools/intervene.py`
加了 random 对照组。两份文档：`docs/2026-11-16_b项_多步预测.md`、
`docs/2026-11-16_e项_目标导向.md`。

### 两个方法论教训（以后别再犯）

1. **时间切分不能用。** 她的世界一直在长新区域：前 60% 拟合、后 40% 打分时，
   测试段 95% 的起点是没见过的。必须前向评估。
2. **基线要取强的那个。** `same` 用先验频率时弱成 5.08 bits，换成在线同名频率是
   2.16——判据的结论完全反过来。这正是 §4 第 2 条要求的那一问。

### 没做（按优先级）

1. 动作判据换成本地量（会改变她之后的轨迹，替代判据要单独设计）。
2. 性能 24.5 ms/步（热点：`drive.deflection` 55%，`terrain.sigma` 1738 次/步）。
3. d 项组合泛化。
4. `SMOOTHING` 不动——试过六种替换，都翻不过「不动」基线，不值得改她的轨迹。
