"""验证代谢：不均匀、不按价值淘汰、随机起伏。"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.L5_cognition.terrain import Terrain

D = 128
rng = np.random.default_rng(42)

terr = Terrain(dim=D, recover_rate=0.02, max_traces=64,
               ceiling=1.0, deposit_ratio=0.05,
               recover_jitter=0.5, decay_spread=0.6)

# 造一批痕迹
for i in range(30):
    x = rng.normal(0, 2.0, D)
    terr.pass_through(x, t_event=None)

traces = terr.traces()
print(f"痕迹数={len(traces)}")

# 1. 每个痕迹有自己的衰减速因子
decays = np.array([t._sigma_decay for t in traces])
print(f"1. 个体衰减速因子: mean={decays.mean():.3f} std={decays.std():.3f} "
      f"min={decays.min():.3f} max={decays.max():.3f}")
assert decays.std() > 0.1, "个体差异应当存在"

# 2. 一次代谢后，各自褪去的比例不同
before = np.array([t.amp for t in traces])
terr.recover(rng=np.random.default_rng(7))
after = np.array([t.amp for t in terr.traces()])
ratios = after / before
print(f"2. 一次代谢的褪去比例: mean={ratios.mean():.4f} std={ratios.std():.4f} "
      f"min={ratios.min():.4f} max={ratios.max():.4f}")
assert ratios.std() > 1e-4, "代谢不应均匀"
assert not np.allclose(ratios, ratios[0]), "各痕迹不应同步褪"

# 3. 不按价值淘汰：浅的不是优先被杀
terr2 = Terrain(dim=D, recover_rate=0.001, max_traces=10, ceiling=1.0, deposit_ratio=0.05)
for i in range(20):
    terr2.pass_through(rng.normal(0, 5.0, D))
print(f"3. 上限 10 但造了 20 条：实际痕迹={len(terr2)}（不再按价值剔除）")
assert len(terr2) > 10, "不应按价值强制裁到上限"

# 4. 长期代谢会自然清掉淡掉的
terr3 = Terrain(dim=D, recover_rate=0.05, max_traces=64, min_amp=1e-3,
                ceiling=1.0, deposit_ratio=0.05)
for i in range(10):
    terr3.pass_through(rng.normal(0, 3.0, D))
n0 = len(terr3)
for _ in range(200):
    terr3.recover(rng=np.random.default_rng(1))
print(f"4. 代谢 200 次：{n0} 条 → {len(terr3)} 条")
assert len(terr3) < n0, "淡掉的痕迹应当消失"

print("\n代谢验证通过")
