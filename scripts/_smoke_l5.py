"""L5 地形验证（快速版）：确认痕迹能形成、地形能凹下去、能看到转移。"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from src.L5_cognition.stream import Stream
import dataclasses

D = 128
stream = Stream(dim=D, gamma=0.05, dt=0.01, recover_rate=0.002, max_traces=256)

rng = np.random.default_rng(42)
stream.calibrate(rng.normal(0.5, 0.1, size=(200, D)))
stream.landscape.set_noise_amplitude(0.02)

t0 = time.time()
events = []
for i in range(3000):
    ev = stream.step()
    if ev is not None:
        events.append(ev)

elapsed = time.time() - t0
ss = stream.stream_state()
prof = stream.state_space.depth_profile()

print(f"3000 步耗时 {elapsed:.2f}s  ({3000/elapsed:.0f} 步/秒)")
print(f"attractors={ss["n_attractors"]}  salient={ss["n_salient"]}  speed={ss['speed']:.4f}  pot={ss['potential']:.3f}")
if len(prof):
    print(f"深度: max={prof.max():.4f} mean={prof.mean():.4f} min={prof.min():.4f}")
print(f"转移事件={len(events)}")
if events:
    nd = sum(1 for e in events if e.noise_driven)
    print(f"  噪声驱动={nd} ({nd/len(events):.0%})")
    for e in events[:5]:
        print(f"  {e.from_trace} -> {e.to_trace} speed={e.speed_before:.4f} "
              f"aligned={e.displacement_aligned:+.2f} noise={e.noise_driven}")

syms = stream.get_symbols()
fields = {f.name for f in dataclasses.fields(syms)}
assert fields == {"trace_id", "transition_seq", "intent", "t_event"}, "符号集泄漏了额外字段"
print(f"\nsymbols: trace={syms.trace_id} intent={syms.intent}")
print("OK：L6 只能看到离散符号")
