r"""通道消融审计（2026-11-28）。

问题：L4 的几条通道（驱动偏转、调节场）到底对她的行为有没有影响？
只读副本：每个模式从同一份存档起跑，同 RNG；用 monkeypatch 关掉通道，比行为。

模式：
- baseline：原样
- no_drive：关掉 drive.deflection（set_drive(None)）
- no_pull：关掉 regulation.factors（set_pull(None)）
- no_both

量：动作序列与 baseline 的差异（Hamming）、leave 占比、区域变化、平均位移、
张力 sum 的均值/CV。

Run: .\.venv\Scripts\python.exe tools\channel_ablation_audit.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.common import persist  # noqa: E402
from src.L5_cognition.stream import Stream  # noqa: E402

STEPS = int(os.environ.get("SHIRO_AUDIT_STEPS", "400"))
FIXED = Path("data/state/_audit_ablate_fixed.pkl")


def run(mode: str):
    cp = Path("data/state/_audit_ablate.pkl")
    shutil.copyfile(FIXED, cp)
    st = Stream()
    assert persist.load_l5(st, name=cp.name)
    if mode in ("no_drive", "no_both"):
        st.drive.deflection = lambda *a, **k: None
    if mode in ("no_pull", "no_both"):
        st.regulation.factors = lambda ss: None
    if mode == "no_recall":
        st.recaller.by_cue = lambda *a, **k: None
    if mode == "no_ast":
        st.attention.observe = lambda *a, **k: None
    if mode == "no_self":
        st.self_model.observe = lambda *a, **k: None
    r0 = len(st.landscape.state_space.attractors())
    actions = []
    sums = []
    disp = []
    trans = 0
    prev_uid = None
    for _ in range(STEPS):
        x0 = np.asarray(st._st.x, dtype=float).copy()
        st.landscape.set_noise_amplitude(0.1957)
        st.step()
        actions.append(int(st._action))
        disp.append(float(np.linalg.norm(np.asarray(st._st.x, dtype=float) - x0)))
        sums.append(float(st.drive.summary().get("sum", 0.0)))
        uid = st.detector.current_attractor
        if prev_uid is not None and uid != prev_uid:
            trans += 1
        prev_uid = uid
    cp.unlink()
    sums_arr = np.asarray(sums, dtype=float)
    return {
        "actions": actions,
        "leave_share": round(float(np.mean(actions)), 3),
        "regions_delta": len(st.landscape.state_space.attractors()) - r0,
        "disp_mean": round(float(np.mean(disp)), 2),
        "sum_mean": round(float(sums_arr.mean()), 1),
        "sum_cv": round(float(sums_arr.std() / max(abs(sums_arr.mean()), 1e-12)), 6),
        "transitions": trans,
    }


def main() -> int:
    if not FIXED.exists():
        shutil.copyfile(Path("data/state/l5.pkl"), FIXED)
    base = run("baseline")
    b_actions = base["actions"]
    out = {"steps": STEPS, "baseline": {k: v for k, v in base.items() if k != "actions"}}
    for mode in ("no_drive", "no_pull", "no_both", "no_recall", "no_ast", "no_self"):
        r = run(mode)
        hamming = sum(1 for a, b in zip(b_actions, r["actions"]) if a != b)
        out[mode] = {k: v for k, v in r.items() if k != "actions"}
        out[mode]["action_hamming_vs_baseline"] = hamming
        out[mode]["action_hamming_share"] = round(hamming / STEPS, 3)
    dest = Path("data/exp/channel_ablation.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("== 通道消融（副本 %d 步）==" % STEPS)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, v)
    print(f"  明细: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

