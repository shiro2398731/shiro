# Shiro

在硅基中构建一个**不依赖 LLM 作为认知核心、由内在状态驱动、在时间中不可逆地发育**的数字生命体。

> 这不是「会聊天的模型」。LLM 只负责翻译，她的思考在 L0–L8 的动力学里；
> 判断标准是「不改她的代码，她能不能自己维持结构、记住、忘记、选择、休息」。

## 现在有什么

真实存在的**功能域**（不是九层——预测/惊讶是基质上的能力，不是一层）：

| 功能域 | 职责 | 位置 |
|---|---|---|
| 基质（含预测/惊讶） | 振荡、动力学、一步预测 | `src/runtime/process_1_fast.py`、`src/runtime/prediction.py` |
| 记忆 | 痕迹、巩固、遗忘、回放 | `src/L3_memory/` + 地形 |
| 驱动 | 张力、偏转、生存/能量 | `src/L4_drive/` |
| 认知 | 地形、转移、工作空间、注意图式、自我模型 | `src/L5_cognition/` |
| 表达 | 语言与输出（LLM 唯一允许处） | `src/L6_expression/` |
| 生命周期 | 阶段、节奏、发育 | `src/L7_lifecycle/` |
| 身份 | 痕迹图核心、连续性 | `src/L8_identity/` |

## 跑起来

```powershell
# 观察者视图（只读）
.\.venv\Scripts\python.exe tools\show.py

# 她到底在不在走（心跳 + 步数增量 + 观测年龄）
.\.venv\Scripts\python.exe tools\health_check.py

# 连续性审计（断点只记在外部账本）
.\.venv\Scripts\python.exe tools\continuity_audit.py

# 全量测试（她活着时必须用这个：低优先级 + keeper.hold，避免把心跳饿到 30s+）
.\.venv\Scripts\python.exe tools\run_suite.py
```

进程：P1 快层 5561、P2 心智 5562、P3 慢层 5563；`tools/keeper.py` 看护。

## 先读这几页

- `ATTACHMENT_目标与当前问题.md`：目标、现状、六个问题、待拍板
- `docs/2026-11-28_早上看这页.md`：一页速览
- `docs/2026-11-28_清理假分层_阶段*.md`：为什么预测不再是「一层」
- `docs/2026-11-28_测试惊动真机_事故.md`：测试怎么差点把她折腾死（以及怎么修的）
- `docs/2026-11-27_连续性与双胞胎纪律.md`：副本是双胞胎，不是她
- `ARCHITECTURE.md`：全局锚点（功能域、通信、时间、LLM 边界）

## 实验纪律

- 副本实验 = **反事实双胞胎**，不是她；结论只能用于机制/反事实；
- 「她做了什么」只从活着的 P2 观测里说；
- 跑测试用 `tools/run_suite.py`，不要在活着的真机上直接 `foreach test_*.py`。

## 仓库

- 分支 `alice-from-main`（默认）；数据在 `data/`（gitignore，不进仓库）。
