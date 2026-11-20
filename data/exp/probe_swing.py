"""副本：接上 swing（给方向），看和只接 rest 有什么不同。

swing = 全局慢向量 s，地形沿它倾斜，走够了反向 → 她"扫"过空间。
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\neolaunchertest\3\bot-3021443116\shiro_alice")
sys.path.insert(0, str(ROOT))
from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 800


def run(use_swing: bool, seed: int = 23) -> dict:
    st = Stream()
    assert persist.load_l5(st)
    st.rng = np.random.default_rng(seed)
    st.landscape._rng = np.random.default_rng(seed)
    st._use_swing = use_swing
    sp = st.landscape.state_space
    n0 = len(sp.attractors())
    seq = []
    for _ in range(STEPS):
        st.landscape.set_noise_amplitude(0.02)
        st.step()
        u = getattr(st.detector, "current_attractor", None)
        if u is not None:
            seq.append(int(u))
    attrs = sp.attractors()
    amps = np.array([a.amp for a in attrs])
    return {
        "use_swing": use_swing,
        "d_regions": len(attrs) - n0,
        "distinct": len(set(seq)),
        "transitions": len(seq) - 1 if seq else 0,
        "amp_gt_01": int((amps > 0.01).sum()),
        "swing_turns": st.swing.turns,
    }


for sw in (False, True):
    print(run(sw))
