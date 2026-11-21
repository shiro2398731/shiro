"""副本：验证「补充」（energy 会回）。"""

import sys
from pathlib import Path

ROOT = Path(r"C:\neolaunchertest\3\bot-3021443116\shiro_alice")
sys.path.insert(0, str(ROOT))
from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

st = Stream()
assert persist.load_l5(st)
print("加载: energy=%.6f burn=%.1e recharge=%.2e asleep=%s" % (
    st.energy, st.energy_burn, st.energy_recharge, st.asleep))

# 1) 从低电量开始，看它会不会回
st.energy = 0.5
e0 = st.energy
leaves = 0
for _ in range(500):
    st.landscape.set_noise_amplitude(0.02)
    if st._action == "leave":
        leaves += 1
    st.step()
print("从 0.5 起跑 500 步: energy %.6f -> %.6f (leave %d 次)" % (e0, st.energy, leaves))

# 2) 满电时封顶
st.energy = 1.0
for _ in range(100):
    st.landscape.set_noise_amplitude(0.02)
    st.step()
print("满电跑 100 步: energy=%.6f（应该 <= 1.0）" % st.energy)

# 3) 存档往返
payload = persist.build_l5_payload(st)
print("payload 有 energy_recharge:", "energy_recharge" in payload)
st2 = Stream()
persist.apply_l5_payload(st2, payload)
print("apply 后: recharge=%.2e" % st2.energy_recharge)
