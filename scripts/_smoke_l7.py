"""L7 验证：见证、阶段、稳态。确认它们只观测不干预。"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.L7_lifecycle.rhythm import PhaseWitness
from src.L7_lifecycle.stages import StageMonitor
from src.common.homeostasis import Homeostasis

# ---------- 1. 相位见证 ----------
w = PhaseWitness()
t0 = 0.0
for i in range(100):
    ts = f"2026-11-03T10:00:{i:02d}.000000+00:00" if i < 60 else f"2026-11-03T10:0{i//60}:{i%60:02d}.000000+00:00"
    w.see("up" if i % 2 == 0 else "down", 0.85, ts)
print(f"1. 相位见证: events={w.count} current={w.current_phase} median_period={w.median_period()}")
assert w.count == 100 and w.current_phase == "down"

# ---------- 2. 阶段 ----------
s = StageMonitor()
# 一开始什么都不满足
for i in range(50):
    s.observe(dev=1.0, s_avg=0.05, n_basins=0, noise_driven_ratio=1.0)
print(f"2a. 样本不足阶段={s.evaluate('t0')}")

# 造一个"dev 有波动 + 学习可见"的序列
for i in range(200):
    s.observe(dev=1.0 + 0.3 * np.sin(i / 10), s_avg=0.05 * (1 - i / 400), n_basins=5, noise_driven_ratio=0.5)
idx = s.evaluate("t1")
print(f"2b. 数据充足后阶段={idx} ({s.stage_def().name})")
assert idx >= 1, "应该至少到 infant"

# 阶段不倒退
s.observe(dev=0.0, s_avg=99.0, n_basins=0, noise_driven_ratio=1.0)
idx2 = s.evaluate("t2")
assert idx2 >= idx, "阶段不应倒退"
print(f"2c. 输入变差后阶段={idx2}（不倒退）OK")

# ---------- 3. 稳态：闭环 ----------
h = Homeostasis()
# 正常闭环：累积高时活动低（负相关）
rng = np.random.default_rng(0)
for i in range(200):
    p = rng.random()
    a = 1.0 - p + rng.normal(0, 0.05)
    h.observe(pressure=p, activity=a, dev=1.0 + rng.normal(0, 0.2))
rep = h.closure()
print(f"3a. 正常闭环: closed={rep.closed} corr={rep.correlation:.3f}")
assert rep.closed and rep.correlation < 0

# 坏掉的闭环：累积高且不变，活动也不变
h2 = Homeostasis()
for i in range(200):
    h2.observe(pressure=0.95, activity=0.01, dev=1.0)
rep2 = h2.closure()
print(f"3b. 坏掉的闭环: closed={rep2.closed} note={rep2.note}")
assert not rep2.closed, "应该检测到闭环断裂"

# ---------- 4. 数值安全（这个可以修） ----------
arr = np.array([1.0, np.nan, 3.0, np.inf])
fallback = np.array([0.0, 2.0, 0.0, 4.0])
fixed, changed = Homeostasis.repair_numeric(arr, fallback)
print(f"4. 数值修复: {arr} -> {fixed} (修过={changed})")
assert changed and np.all(np.isfinite(fixed))
assert fixed[1] == 2.0 and fixed[3] == 4.0

print("\nL7 全部通过")
