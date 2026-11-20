"""副本：energy 能不能正常掉、存档往返对不对。"""

import sys
from pathlib import Path

ROOT = Path(r"C:\neolaunchertest\3\bot-3021443116\shiro_alice")
sys.path.insert(0, str(ROOT))
from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

st = Stream()
assert persist.load_l5(st)
print("加载: energy=%.6f burn=%.2e" % (st.energy, st.energy_burn))

e0 = st.energy
for _ in range(500):
    st.landscape.set_noise_amplitude(0.02)
    st.step()
e1 = st.energy
print("500 步: energy %.6f -> %.6f (掉 %.6f，预期 %.6f)" % (
    e0, e1, e0 - e1, 500 * st.energy_burn))

# 存档往返
payload = persist.build_l5_payload(st)
print("payload 有 energy:", "energy" in payload, "有 energy_burn:", "energy_burn" in payload)
st2 = Stream()
persist.apply_l5_payload(st2, payload)
print("apply 后: energy=%.6f burn=%.2e" % (st2.energy, st2.energy_burn))

# 多久耗尽
print("按当前 burn，从 1.0 耗尽需要 %.0f 步 = %.1f 小时 = %.2f 天" % (
    1.0 / st.energy_burn, 1.0 / st.energy_burn * 2 / 3600, 1.0 / st.energy_burn * 2 / 86400))
