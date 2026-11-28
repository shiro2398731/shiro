r"""极性原型：内驱做主（副本，不碰真机）。

模式（同一固定存档、同 RNG、500 步）：
- baseline：现状（张力 sum 阈值 + 最近未去过/回家）
- drive / drive_home：偏转直接做主（后者离家>4σ 时混入回家方向 0.3）
- gated_drive：闸保留（张力阈值），只把「出门方向」换成偏转
- gated_local_drive：闸保留，在走得到的候选里挑最顺偏转的
- gated_pair_drive：闸保留，走向「当前区域参与的张力最大 pair」的另一端

量：到达区域、最大位移、结构变化、点火、位移、偏转-位移余弦。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.action import HOLD, LEAVE  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = 500
FIXED = Path("data/state/_audit_polarity_fixed.pkl")


def cosine(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def _drive_unit(st):
    d = st.drive.deflection(st.landscape.state_space, st.graph)
    if d is None:
        return None
    d = np.asarray(d, dtype=float)
    n = float(np.linalg.norm(d))
    return d / n if n > 0 else None


def _home_push(st, x, scale, weight=0.3):
    if st._home_x is None:
        return np.zeros_like(x)
    hv = np.asarray(st._home_x) - x
    hd = float(np.linalg.norm(hv))
    if hd > float(st.home_radius_sigmas) * scale:
        return weight * (hv / max(hd, 1e-12))
    return np.zeros_like(x)


def run(mode: str):
    cp = Path("data/state/_audit_polarity.pkl")
    shutil.copyfile(FIXED, cp)
    st = Stream()
    assert persist.load_l5(st, name=cp.name)

    if mode in ("drive", "drive_home"):
        def choose():
            st._tension_now = float(st.drive.summary().get("sum", 0.0))
            st._tension_history.append(st._tension_now)
            u = _drive_unit(st)
            if u is None:
                return HOLD
            scale = float(st._own_scale() or 100.0)
            if mode == "drive_home":
                u = u + _home_push(st, np.asarray(st._st.x), scale)
                u = u / max(float(np.linalg.norm(u)), 1e-12)
            st._st = type(st._st)(x=np.asarray(st._st.x) + u * scale, v=st._st.v)
            st.action_counts[LEAVE] = st.action_counts.get(LEAVE, 0) + 1
            return LEAVE
        st._choose_action = choose

    elif mode in ("gated_drive", "gated_local_drive", "gated_pair_drive"):
        def apply_leave():
            attrs = st.landscape.state_space.attractors()
            if not attrs:
                return
            x = np.asarray(st._st.x, dtype=float)
            scale = float(st._own_scale() or 100.0)
            u = _drive_unit(st)
            if mode == "gated_drive":
                if u is None:
                    return
                u = u + _home_push(st, x, scale)
                step = scale
            elif mode == "gated_local_drive":
                if u is None:
                    return
                best = None
                for a in attrs:
                    mu = np.asarray(a.mu, dtype=float)
                    dv = mu - x
                    dd = float(np.linalg.norm(dv))
                    if dd <= 0.5 * scale or dd > 4.0 * scale:
                        continue
                    cand = dv / dd
                    s = float(u @ cand) + float(_home_push(st, x, scale) @ cand)
                    if best is None or s > best[0]:
                        best = (s, cand, dd)
                if best is None:
                    return
                u, step = best[1], min(scale, best[2])
            else:  # gated_pair_drive
                cur = st.detector.current_attractor
                uid_row = st.landscape.state_space.uid_row()
                if cur is None or int(cur) not in uid_row:
                    return
                ts = sorted(st.drive.last, key=lambda t: t.amount, reverse=True)
                target = None
                for t in ts:
                    if int(t.a) == int(cur):
                        target = int(t.b)
                        break
                    if int(t.b) == int(cur):
                        target = int(t.a)
                        break
                if target is None or target not in uid_row:
                    return
                mu = np.asarray(attrs[uid_row[target]].mu, dtype=float)
                dv = mu - x
                dd = float(np.linalg.norm(dv))
                if dd <= 1e-12:
                    return
                u = dv / dd + _home_push(st, x, scale)
                u = u / max(float(np.linalg.norm(u)), 1e-12)
                step = min(scale, dd)
            u = u / max(float(np.linalg.norm(u)), 1e-12)
            st._st = type(st._st)(x=x + u * step, v=st._st.v)
        st._apply_leave = apply_leave

    x0 = np.asarray(st._st.x, dtype=float).copy()
    r0 = len(st.landscape.state_space.attractors())
    seen = set()
    far = 0.0
    disp = []
    ign0 = int(st.workspace.total_ignitions)
    cos = []
    for _ in range(STEPS):
        drive = getattr(st.landscape, "_drive", None)
        dv = np.asarray(drive, dtype=float) if drive is not None else np.zeros(1)
        x_before = np.asarray(st._st.x, dtype=float).copy()
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        dx = np.asarray(st._st.x, dtype=float) - x_before
        disp.append(float(np.linalg.norm(dx)))
        cos.append(cosine(dv, dx))
        uid = st.detector.current_attractor
        if uid is not None:
            seen.add(int(uid))
        far = max(far, float(np.linalg.norm(np.asarray(st._st.x) - x0)))
    try:
        cp.unlink()
    except FileNotFoundError:
        pass
    return {
        "regions": len(seen),
        "max_disp_from_start": round(far, 1),
        "regions_delta": len(st.landscape.state_space.attractors()) - r0,
        "ignitions": int(st.workspace.total_ignitions) - ign0,
        "disp_mean": round(float(np.mean(disp)), 1),
        "cos_drive_disp": round(float(np.mean(cos)), 4),
    }


def main() -> int:
    if not FIXED.exists():
        shutil.copyfile(Path("data/state/l5.pkl"), FIXED)
    for mode in ("baseline", "gated_drive", "gated_local_drive", "gated_pair_drive"):
        print(mode, run(mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
