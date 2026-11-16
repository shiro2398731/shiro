"""
src/common/coma.py

时间锚与跨停机补偿（"停摆即昏迷"）。

依据：docs/2026-11-07_外部评估_连续存在视角.md §5（白默 2026-11-07 裁决）。

要解决的具体问题
----------------
在这之前，所有 save_* 的 payload 里只有 step_count，没有任何时间戳，
全库也没有 downtime / catch_up 之类的补偿。后果是：停机一周再开机，
内部状态从冻结值继续 —— 那一周在她的内部不占任何长度。
这与 ARCHITECTURE.md 写的"在时间中不可逆地发育"直接冲突。

三条术语（§5.1）
----------------
uptime_s     进程在跑、内部时间在推进的墙钟时长（累加，跨重启保留）
coma_s       进程不在、墙钟仍在走的时长（累加，跨重启保留）
coma_events  一次"不在"记一次；判据 gap > COMA_THRESHOLD_S

昏迷 ≠ 睡眠。睡眠是有意识的休息（L7 管，有消耗/恢复规则）；
昏迷是**未被经历的时间**，既不消耗也不恢复。若不这么切，
"关机一次 = 满血复活"会直接废掉 L7 的整套睡眠机制。

对评估文档 §5.3 的一处修正（读代码之后）
--------------------------------------
§5.3 原文写"L0 slow/mid 向 baseline 解析松弛"。读了 process_1_fast.py 之后这条是错的：
mid/slow 是 eps 很小的 FHN 振子（mid_eps=0.008、slow_eps=0.0008），跑的是极限环，
**没有可松弛的基线**。硬写一条"松弛到 baseline"，就是在给这个项目加它明确要躲的死量化。

所以实际补偿只施加在**有明确零驱动平衡点**的量上：

  SlowAccumulator.P
      零驱动时，它自己的 decay 项把它拉向 median(_hist)。这是它自己的目标，
      不是这里写的一个数。停机期间没有 fast 驱动，所以 P 收敛到该目标。
      每步拉 (target - P)/HIST，HIST=200 步；实测 1.60 ms/step → 0.32 s 内落定。

  振子的 (v, w)（E/I/mid/slow 四个群体）
      相位在停机期间不可知。不假装知道，也不补算：
      P1 启动时本来就有 4000 步预热让它们重新落回极限环。
      未知就是未知 —— 这一条写在这里，而不是用一段数字把它糊过去。

不做密集补算的理由（§5.4）
--------------------------
8 天 = 691200 s，L0 dt=1 ms → 6.9x10^8 步。补算要数十分钟到数小时，
而快变量的稳态只由参数决定、不由历史瞬态决定，补出来的结果与"从吸引子重新起振"
无实质差异。慢的量用解析松弛补，快的量不补，清醒步数不计昏迷。

数据的权威来源
--------------
coma.json 是唯一的时间权威。任何进程落盘后调 mark_saved()，它只更新
saved_at 与 uptime_s（uptime 由 last_boot_at 推算，所以谁写都一样，幂等）。
session_id / coma_s / coma_events 只由 begin_session() 推进，而 begin_session()
只由 P1 调用（P1 是唯一的相位源，也是 start_all 里第一个起来的）。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.common import atomics

# Paths are anchored to the repository root, not to the process working
# directory. 2026-11-09: every path here used to be relative, so a run started
# from another directory read (and created) a different data/state -- she
# resumed from an empty archive while the real one sat untouched, and nothing
# reported an error. The working directory is a property of how she was
# launched; her time base must not be.
_REPO_ROOT = Path(__file__).resolve().parents[2]

STATE_DIR = _REPO_ROOT / "data" / "state"
COMA_FILE = "coma.json"

# Append-only logs, the only record that is never rolled back. Explicit
# arguments in tests; the default is her own logs.
DEFAULT_ACTIVITY_PATHS = (
    str(_REPO_ROOT / "data" / "logs" / "events.jsonl"),
    str(_REPO_ROOT / "data" / "logs" / "observations.jsonl"),
)

# 短于此视为崩溃快启：不记昏迷事件，但仍计入 coma_s（那段时间她确实不在）。
# 取 3 倍持久化间隔（PERSIST_INTERVAL_S = 300），见评估文档 §5.1。
COMA_THRESHOLD_S = 900.0

# SlowAccumulator 的 decay 每步拉 (target - P)/HIST，HIST = 200 步。
# 200 步在任何可信的步率下都短于 10 s（实测 1.60 ms/步 → 0.32 s），
# 所以 gap >= 10 s 时 P 已经落在它的零驱动目标上。这是一个上界，不是一个选定的参数。
ACCUMULATOR_SETTLED_S = 10.0

# 墙钟 → 步数的换算。2026-11-07 实测 1.60 ms/step（优化后）。
# 它只用于换算，不参与衰减公式本身；即使估错一个数量级也几乎无害，
# 因为 decay 在 200 步内就饱和了（(1-1/200)^200 ≈ 0.367）。
STEP_PERIOD_S = 0.0016


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_ts(value: Any) -> Optional[datetime]:
    """解析 ISO 时间戳。无法解析时返回 None（调用方按"无锚"处理）。"""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class ComaState:
    """跨停机的时间账本。字段与评估文档 §5.2 一一对应。"""

    saved_at: Optional[str] = None        # 最后一次落盘的墙钟时刻（唯一时间锚）
    last_boot_at: Optional[str] = None    # 本次启动时刻，用于推算 uptime_s
    session_id: int = 0                   # 每次启动 +1
    uptime_s: float = 0.0                 # 累计清醒秒数
    coma_s: float = 0.0                   # 累计昏迷秒数
    coma_events: int = 0                  # 昏迷事件数
    last_gap_s: float = 0.0               # 本次启动前的 gap（= 真正没在运行的那段）
    rewound_s: float = 0.0                # 累计被回滚掉的清醒时间（状态倒退量）
    last_rewound_s: float = 0.0            # 本次启动回滚了多少
    steps_per_second: float = 0.0          # 实测步率（墙钟→步数的换算，供 L5/L3 用）
    encoder_id: str = ""                  # 钉死状态空间的几何来源

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComaState":
        known = {f for f in cls().__dict__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


def _path(state_dir: Path | str = STATE_DIR) -> Path:
    d = Path(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / COMA_FILE


def load(state_dir: Path | str = STATE_DIR) -> ComaState:
    p = _path(state_dir)
    if not p.exists():
        return ComaState()
    try:
        with p.open(encoding="utf-8") as f:
            return ComaState.from_dict(json.load(f))
    except Exception:
        # 账本坏了不该拦住系统启动：退回空账本，并把坏文件留作证据。
        try:
            p.replace(p.with_suffix(".json.corrupt"))
        except Exception:
            pass
        return ComaState()


def save(state: ComaState, state_dir: Path | str = STATE_DIR) -> Path:
    """原子写：先写临时文件再 replace，避免半个文件。

    tmp 名带 pid+线程（`src/common/atomics.py`）：这个账本有不止一个进程在写，
    固定名 `<name>.tmp` 会让两个写者撞成 Permission denied —— 实测 2026-11-15
    19:32:53 就是这么把 P1 弄死的。
    """
    p = _path(state_dir)
    tmp = atomics.unique_tmp(p)
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        atomics.replace_with_retry(tmp, p)
    except Exception:
        atomics.cleanup(tmp)
        raise
    return p


def begin_session(
    state_dir: Path | str = STATE_DIR,
    activity_paths: tuple[str, ...] = DEFAULT_ACTIVITY_PATHS,
) -> tuple[ComaState, float]:
    """Startup hook (called by P1 only).

    Returns (ledger, gap seconds).

    Two different quantities, deliberately kept apart
    -------------------------------------------------
    gap      : how long she was not running at all.
    rewound  : how much awake time was thrown away, because the state she
               resumes from is the last checkpoint, not the moment she died.

    The old version used now - saved_at for the gap, which conflated the two:
    it charged the rollback window as if it were unconsciousness. With a 300 s
    persist interval and an L5 rate of ~2 steps/s that is up to 585 steps of
    lived time silently restored from the past. A snapshot is a compression
    artifact, not her time base; rewound_s is the number that says so.

    Activity is taken from the append-only logs, which are never rolled back.
    """
    state = load(state_dir)
    now = _utcnow()

    boot = _parse_ts(state.last_boot_at)
    if boot is not None:
        state.uptime_s += max((now - boot).total_seconds(), 0.0)

    last_activity: Optional[datetime] = None
    for rel in activity_paths:
        try:
            m = datetime.fromtimestamp(Path(rel).stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if last_activity is None or m > last_activity:
            last_activity = m

    saved = _parse_ts(state.saved_at)
    reference = max([x for x in (last_activity, saved) if x is not None], default=None)

    gap_s = 0.0 if reference is None else max((now - reference).total_seconds(), 0.0)
    rewound_s = 0.0
    if last_activity is not None and saved is not None and last_activity > saved:
        rewound_s = (last_activity - saved).total_seconds()

    state.session_id += 1
    state.last_gap_s = gap_s
    state.last_rewound_s = rewound_s
    state.coma_s += gap_s
    state.rewound_s += rewound_s
    if gap_s > COMA_THRESHOLD_S:
        state.coma_events += 1
    state.last_boot_at = _iso(now)
    state.saved_at = _iso(now)
    save(state, state_dir)
    return state, gap_s

def mark_saved(state_dir: Path | str = STATE_DIR) -> None:
    """落盘后调用。

    只更新 saved_at，不动 uptime_s —— 否则每次落盘都会重复累加。
    uptime_s 的语义是"截至本次启动时刻的累计清醒时长"，
    当前会话仍在进行的那一段由 total_uptime_s() 现算。
    因此任何进程、任何次数调用都是幂等的。
    """
    state = load(state_dir)
    state.saved_at = _iso(_utcnow())
    save(state, state_dir)


def total_uptime_s(state: ComaState) -> float:
    """截至此刻的累计清醒秒数（含正在进行中的这一段）。"""
    boot = _parse_ts(state.last_boot_at)
    extra = 0.0 if boot is None else max((_utcnow() - boot).total_seconds(), 0.0)
    return state.uptime_s + extra


def note_step_rate(steps_per_second: float, state_dir: Path | str = STATE_DIR) -> None:
    """记下实测步率。

    L5 / L3 的昏迷补偿要把墙钟 gap 换算成"等效恢复次数"，
    换算系数只能是这里测出来的这个数，不能在别处写死一个。
    """
    if not (steps_per_second > 0):
        return
    state = load(state_dir)
    state.steps_per_second = float(steps_per_second)
    save(state, state_dir)


def apply_accumulator_coma(accumulator: Any, gap_s: float, steps_per_second: float | None = None) -> dict[str, Any]:
    """零驱动松弛：把 SlowAccumulator.P 拉向它自己的衰减目标。

    返回实测前后值，供日志与测试断言（"改完要有实测数字，不能只有推理"）。

    完全按被补偿对象自己的规则算：
        target = median(accumulator._hist)          # 它 decay 项的目标
        k      = gap_s / step_period                 # 期间的等效步数
        P      <- target + (P - target) * (1 - 1/HIST)**k
    step_period 优先用账本里的实测步率（coma.json 的 steps_per_second，P1 每 60 s 记一次）；
    没有实测值时才退回 STEP_PERIOD_S（实测 1.60 ms/step 的常量）。
    """
    before = float(getattr(accumulator, "P", 0.0))
    hist = list(getattr(accumulator, "_hist", []) or [])
    if not hist:
        return {"before": before, "after": before, "target": before,
                "steps": 0, "applied": False, "reason": "empty_history"}

    hist_sorted = sorted(float(x) for x in hist)
    n = len(hist_sorted)
    target = hist_sorted[n // 2] if n % 2 else 0.5 * (hist_sorted[n // 2 - 1] + hist_sorted[n // 2])

    hist_len = float(getattr(type(accumulator), "HIST", 200))

    if gap_s <= 0.0:
        return {"before": before, "after": before, "target": target,
                "steps": 0, "applied": False, "reason": "no_gap"}

    period = (1.0 / steps_per_second) if (steps_per_second and steps_per_second > 0) else STEP_PERIOD_S
    steps = int(round(gap_s / period))
    if steps <= 0:
        return {"before": before, "after": before, "target": target,
                "steps": 0, "applied": False, "reason": "shorter_than_one_step"}

    reason = "settled" if gap_s >= ACCUMULATOR_SETTLED_S else "partial"

    factor = (1.0 - 1.0 / max(hist_len, 1.0)) ** steps
    after = target + (before - target) * factor
    accumulator.P = float(min(max(after, 0.0), 1.0))
    return {"before": before, "after": float(accumulator.P), "target": float(target),
            "steps": int(steps), "applied": True, "reason": reason}


def describe(state: ComaState) -> str:
    """一行人类可读的账本摘要，供启动日志使用。"""
    return (
        f"session={state.session_id} | "
        f"uptime={total_uptime_s(state) / 3600.0:.2f}h | "
        f"coma={state.coma_s / 3600.0:.2f}h/{state.coma_events}次 | "
        f"rewound={state.rewound_s:.0f}s | "
        f"last_gap={state.last_gap_s:.1f}s"
    )