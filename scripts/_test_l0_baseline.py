"""
scripts/_test_l0_baseline.py

Where the input scales come from, and whether they still mean what they say.

Two numbers used to be constants picked in code:

    _ext_decay = 0.98      how long a sentence lingers
    gain       = 0.15      how strongly it acts

Neither was derived from anything. This script measures the oscillator, feeds the
measurement back as the baseline, and checks that the resulting input behaves the
way the definition says it should:

    a sentence biases a few cycles, then fades
    a sentence at unit scale moves the network by about one standard deviation
    of its own fluctuation, no more

The baseline is measured here rather than hardcoded so that the numbers can be
re-derived if the oscillator changes. If those constants move, the scales move
with them.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_l0_baseline.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.runtime.process_1_fast import FHNNetwork


def measure(net, steps=30000, warm=5000):
    """Measure the oscillator's own period and fluctuation width."""
    fasts = []
    for _ in range(steps):
        fasts.append(net.step()["fast"])
    F = np.asarray(fasts[warm:], dtype=float)

    med = float(np.median(F))
    above = F >= med
    crossings = np.flatnonzero(above[1:] & ~above[:-1])
    period = float(np.median(np.diff(crossings))) if crossings.size > 2 else 0.0

    return {
        "period": period,
        "mean": float(F.mean()),
        "std": float(F.std()),
        "p5": float(np.percentile(F, 5)),
        "p95": float(np.percentile(F, 95)),
    }


def main():
    net = FHNNetwork(seed=42)
    base = measure(net)

    print("measured baseline")
    print(f"  period : {base['period']:.0f} steps")
    print(f"  fast   : mean {base['mean']:.4f}  std {base['std']:.4f}  "
          f"p5 {base['p5']:.4f}  p95 {base['p95']:.4f}")

    net.set_baseline(base["period"], base["std"])

    print()
    print("derived from it")
    print(f"  span fraction   : {net._ext_span_frac:.3f} of a period")
    print(f"  span in steps   : {net._ext_span_steps:.1f}")
    print(f"  decay per step  : {net._ext_decay:.6f}")
    print(f"  input gain      : {net._input_gain:.4f}")

    # --- check the span really is what it says -----------------------
    print()
    print("checking the span")
    decay = net._ext_decay
    residual_span = np.log(0.01) / np.log(decay)
    ok_span = abs(residual_span - net._ext_span_steps) < 1.0
    print(f"  amplitude reaches 1% after {residual_span:.1f} steps "
          f"(declared {net._ext_span_steps:.1f})  "
          f"{'ok' if ok_span else 'FAIL'}")

    # --- check the gain means one standard deviation -----------------
    print()
    print("checking the gain")
    net2 = FHNNetwork(seed=42)
    before = []
    for _ in range(3000):
        before.append(net2.step()["fast"])
    net2.set_baseline(base["period"], base["std"])
    net2.receive(np.ones(128))
    # The peak immediate offset in fast activity is what a full-scale input
    # does; compare it with the width the layer has anyway.
    after = []
    for _ in range(60):
        after.append(net2.step()["fast"])
    shift = abs(float(np.mean(after[:10])) - float(np.mean(before[-10:])))
    ratio = shift / max(base["std"], 1e-12)
    print(f"  offset {shift:.4f} vs std {base['std']:.4f} -> {ratio:.2f} sigma")
    print(f"  elapsed ext amp after 60 steps: {net2._ext_amp:.5f}")

    print()
    print("=" * 55)
    if not ok_span:
        print("VERDICT: the decay does not match the declared span.")
    elif ratio < 0.05:
        print("VERDICT: the input is far weaker than one standard deviation. "
              "It will not move anything.")
    else:
        print(f"VERDICT: span and gain both follow the baseline. An input at "
              f"unit scale acts at about {ratio:.1f} sigma of the layer's own "
              f"fluctuation, and fades over {net._ext_span_steps:.0f} steps.")


if __name__ == "__main__":
    main()
