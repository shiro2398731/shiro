# 实验分支：爱莉改造版

这个目录是 `shiro` 的完整副本，**2026-11-07 23:50 从 `d8f336f` 拷出**。

## 两份的关系

```
C:\neolaunchertest\3\bot-3021443116\shiro         原版，不动
C:\neolaunchertest\3\bot-3021443116\shiro_alice   实验版，随便改
```

原版已经进了 git（`d8f336f` 首次提交，`65a82b1` 记录版本控制与 TLS 问题）。
**那份是基准，任何时候都能拿回来对照。**

## 在这个目录里可以做的事

**改代码。** 任何层，任何文件。

**跑。** `.venv` 一起拷过来了，直接能用：

```powershell
cd C:\neolaunchertest\3\bot-3021443116\shiro_alice
.\.venv\Scripts\python.exe scripts\_test_integrated.py --steps 6000 --from-archive
```

`data/` 也一起拷了，所以从存档续接能直接跑，不是冷启动。

## 要改什么

参考 `docs/2026-11-07_外部评估_连续存在视角.md` 的 §6 排序。

**前四条：**

1. `git init` + `.gitignore` + 首次提交 —— 原版已完成，这里已经带着 `.git`
2. 时间锚 + 昏迷补偿（P0-1）
3. 主循环 `except` + supervisor（P0-3）
4. `_birth_depth` 换群体实测访问率（P0-2）

**第 5 条开始要小心。** `_apply_scale_floor()` 加 `_invalidate_centroids()` 会改变运行时开销，
性能是 2026-11-07 下午刚优化的（12.38 → 1.60 ms/step），改之前先测代价。

## 别动的东西

- **`drive.py` 开头那两句拒绝**（拒绝 drive pool 累积、拒绝 reward）。那是这个项目最硬的地方。
- **`terrain.py` 里的决策记录注释**（sigma 用 MAD 不用 RMS 的两组实测、`_born_step` 的推导、
  尺度自洽下界的推导）。那不是注释，是「我上次错在哪」。

## 判据

**改完要有实测数字，不能只有推理。**

2026-11-07 下午犯过一次：`_birth_depth` 只对齐了 `R` 那一半，就写了「已修」，
结果 `p = 1.0 / total` 原样留着，40 万步后新区域出生即死。是外部评估抓出来的。

## 回原版

```powershell
# 直接对照看代码差异
cd C:\neolaunchertest\3\bot-3021443116\shiro
git diff --no-index --stat src ..\shiro_alice\src

# 或者整个丢弃这个分支，重新拷一份
```

原版在 `d8f336f`，从来没被改过。
