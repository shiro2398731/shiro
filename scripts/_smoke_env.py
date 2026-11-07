"""
开发期单元验证（不是数据生产）。

  1. SlowAccumulator 的不对称性：涨得快、落得慢
  2. suppression_gate 是留给 L4 的空位，现在恒为 1.0，但可被设置
  3. EnvelopeDetector 在合成方波上能正确读出周期（时间窗口解耦）

运行：.venv\\Scripts\\python.exe scripts\\_smoke_env.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from src.runtime.process_1_fast import SlowAccumulator, EnvelopeDetector


def test_accumulator_asymmetry():
    """涨得快、落得慢。观测窗口必须远小于 tau，否则两端都饱和。"""
    DT = 0.001
    STEPS = 2000          # 2 秒

    up = SlowAccumulator(tau_up_s=10.0, tau_down_s=20.0, dt_s=DT)
    for _ in range(50):
        up.step(0.5)      # 先把历史填满
    base = up.P
    for _ in range(STEPS):
        up.step(1.0)      # 高于自身中位数
    rise = up.P - base

    down = SlowAccumulator(tau_up_s=10.0, tau_down_s=20.0, dt_s=DT)
    for _ in range(50):
        down.step(0.5)
    down.P = 1.0
    for _ in range(STEPS):
        down.step(0.0)    # 低于自身中位数
    fall = 1.0 - down.P

    print(f"[accum] 2s 上升={rise:.5f}  2s 下降={fall:.5f}")
    assert rise > fall, f"应涨得快落得慢（rise={rise}, fall={fall}）"
    print(f"[accum] OK：上升/下降 = {rise / fall:.2f}")


def test_accumulator_no_binary():
    """连续累积：中等活动也应有微小变化，不是非黑即白。"""
    acc = SlowAccumulator(tau_up_s=10.0, tau_down_s=20.0, dt_s=0.001)
    for _ in range(50):
        acc.step(0.5)
    p0 = acc.P
    for _ in range(2000):
        acc.step(0.55)    # 只比中位数高一点点
    delta = acc.P - p0
    print(f"[accum] 轻微偏离 2s 后变化 = {delta:.6f}")
    assert delta > 0, "高于中位数即应上升"
    print("[accum] OK：连续响应，无二分")


def test_gate():
    """留给 L4 的空位：现在恒为 1.0，但可被覆盖。"""
    acc = SlowAccumulator(tau_up_s=10.0, tau_down_s=20.0, dt_s=0.001)
    for _ in range(100):
        acc.step(0.9)
    print(f"[gate] P={acc.P:.4f} expressed={acc.expressed:.4f} gate={acc.suppression_gate}")
    assert acc.suppression_gate == 1.0
    assert abs(acc.expressed - acc.P) < 1e-9
    acc.suppression_gate = 0.2
    assert abs(acc.expressed - acc.P * 0.2) < 1e-9
    print("[gate] OK：空位可被覆盖，当前无影响")


def test_detector_scale():
    """检测器读出的是输入信号周期，不是自身采样间隔。"""
    det = EnvelopeDetector()
    t = 0.0
    dt = 0.001
    period = 0.35        # 接近 L0 实测周期
    for i in range(int(6 / dt)):
        A = 1.0 if (i * dt) % period < period / 2 else 0.0
        t += dt
        det.update(A, t)
    est = det._period()
    print(f"[detector] 合成周期=0.35s 估计={est}")
    assert est is not None and 0.28 < est < 0.45, f"周期估计离谱：{est}"
    print("[detector] OK：读的是信号周期")


if __name__ == "__main__":
    test_accumulator_asymmetry()
    test_accumulator_no_binary()
    test_gate()
    test_detector_scale()
    print("\nall smoke tests passed")
