r"""tools/watch_ignition.py — 订阅 l5.state.stream，打印 ignition（验证 GWT 接 L6 的广播层）。

不需要 API key。跑几秒，看有没有 ignition 字段和它的值。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.bus import Bus  # noqa: E402


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    bus = Bus(process_id="P9")
    seen = {"n": 0, "has_ignition": 0, "ignitions": {}, "last": None}

    def on_stream(topic, s):
        try:
            d = json.loads(s)
        except Exception:  # noqa: BLE001
            return
        seen["n"] += 1
        ign = d.get("ignition", "MISSING")
        if ign != "MISSING":
            seen["has_ignition"] += 1
            seen["ignitions"][str(ign)] = seen["ignitions"].get(str(ign), 0) + 1
        seen["last"] = d

    bus.subscribe("l5.state.stream", on_stream, parser=lambda s: s,
                  owner_module="tools.watch_ignition")
    t0 = time.time()
    while time.time() - t0 < seconds:
        time.sleep(0.2)

    print(f"订阅 {seconds:.0f}s：收到 {seen['n']} 条广播")
    print(f"其中带 ignition 字段：{seen['has_ignition']} 条")
    print(f"ignition 值分布（前 5）：{dict(list(seen['ignitions'].items())[:5])}")
    if seen["last"]:
        print(f"最后一条：{ {k: seen['last'].get(k) for k in ('attractor_id','ignition','potential')} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
