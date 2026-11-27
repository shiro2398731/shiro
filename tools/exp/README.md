# tools/exp —— 双胞胎实验协议

这些脚本都是**反事实双胞胎**实验，不是她。

## 为什么

从 `l5.pkl` 续跑的任何副本有她的过去，但是另一条因果链。
它的结论只能回答「如果换一种决定会怎样」，**不能写成「她做了什么」**。
详见 `docs/2026-11-27_连续性与双胞胎纪律.md`。

## 协议（所有脚本必须遵守）

1. **固定起点**：先 snapshot 一份 `data/state/_delib_fixed.pkl`，所有模式从同一份起跑；
   在输出里打印它的 SHA-256，证明「同一把尺子」。
2. **同一 RNG**：从固定存档加载后 `rng` 状态一致，噪声序列一致；
   模式差异只来自「用哪套决定」。
3. **相同步数**：现在默认 400 步（长跑另说）。
4. **指标固定**：到达区域 / 最大位移 / 结构增长 / 新点火 / leave 占比。
5. **真机只读**：脚本只 copy 存档，不写回、不碰 P1/P2/P3。

## 现有脚本

| 脚本 | 做什么 |
|---|---|
| `probe_deliberation_paired.py` | 原规则 / 推演 v4 / 价值 / 能量 / 多步前瞻，同一存档同 RNG 配对 |
| `probe_deliberation_vs_rules.py` | 早期对照（每版从当时最新存档起跑，**不能横向比**，仅留档） |
| `probe_filter_tune.py` | 候选过滤半径调参（外部旋钮实验，另一条线） |

## 跑之前

```powershell
# 固定起点（只需一次；之后所有模式都读这份）
Copy-Item data\state\l5.pkl data\state\_delib_fixed.pkl
.\.venv\Scripts\python.exe tools\exp\probe_deliberation_paired.py
```

如果 `_delib_fixed.pkl` 不存在，`probe_deliberation_paired.py` 会自己 snapshot 一份。
