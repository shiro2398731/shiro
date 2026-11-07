"""
scripts/_test_l0_metabolism.py

Does structural change in L0 stay slow enough to keep it an oscillator?

Adding metabolism to the base layer is not free. The oscillation is produced by
a balance between excitatory coupling and inhibitory drive; if the weights drift
the balance can tip, and the layer that everything else is built on stops
oscillating.

So this measures two things at once:

    drift   how far the coupling weights have moved from their initial values
    period  whether the oscillation is still there, and at what period

A run where drift grows but the period stays near 347 is a layer that ages
without dying. A run where the period wanders or the amplitude collapses is a
layer that metabolism has broken.

The activity scale is supplied from `pressure`, the oscillator's own accumulated
activity: structure changes faster when more is happening. That is the point of
making the rate adjustable, so it is exercised here rather than tested
separately.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_l0_metabolism.py --steps 60000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.runtime.process_1_fast import FHNNetwork, EnvelopeDetector


def relative_drift(net) -> dict:
    """Mean |current - initial| / mean |initial|, per matrix."""
    out = {}
    for name, m in net._coupling_matrices().items():
        base = net.metabolism._base.get(name)
        if base is None:
            continue
        denom = max(float(np.abs(base).mean()), 1e-12)
        out[name] = float(np.abs(m - base).mean() / denom)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--every", type=int, default=10000)
    ap.add_argument("--no-metabolism", action="store_true",
                    help="control run: structure frozen")
    ap.add_argument("--scale", default="fast",
                    choices=("fast", "pressure", "unit"),
                    help="what modulates the metabolic rate")
    args = ap.parse_args()

    net = FHNNetwork(seed=42)

    # The envelope detector measures periods in wall-clock time, which is
    # meaningless here: these steps run as fast as the CPU allows, so its
    # sampling window never fills. The period is instead counted in steps, by
    # finding upward crossings of a slow envelope of the fast variable against
    # that envelope's own running median. Crossing the raw signal's median
    # catches every wobble and reported a period of 1 step.
    env = []
    crossings = []
    was_above = None
    last_cross = None

    print(f"{'step':>7} {'fast':>7} {'scale':>8} "
          f"{'drift_EE':>9} {'drift_EI':>9} {'drift_IE':>9} {'period':>8}")

    for i in range(1, args.steps + 1):
        snap = net.step()
        fast = float(snap["fast"])

        env.append(fast)
        if len(env) > 8:
            env.pop(0)
        if len(env) >= 8:
            smooth = float(np.mean(env))
            ref = float(np.median(env))
            above = smooth >= ref
            if was_above is not None and above and not was_above and last_cross is not None:
                crossings.append(i - last_cross)
                last_cross = i
            elif last_cross is None:
                last_cross = i
            was_above = above

        if not args.no_metabolism:
            if args.scale == "fast":
                scale = fast
            elif args.scale == "pressure":
                scale = float(snap["pressure"])
            else:
                scale = 1.0
            net.metabolize(activity_scale=scale)

        if i % args.every == 0 or i == 1:
            d = relative_drift(net)
            p = crossings[-1] if crossings else None
            print(f"{i:>7} {fast:>7.3f} {net.metabolism._rate_scale:>8.3f} "
                  f"{d.get('EE', 0):>9.5f} {d.get('EI', 0):>9.5f} "
                  f"{d.get('IE', 0):>9.5f} "
                  f"{(f'{p}' if p else '-'):>8}")

    print()
    print("=" * 60)
    d = relative_drift(net)
    print(f"final drift: {({k: round(v, 5) for k, v in d.items()})}")
    if crossings:
        arr = np.asarray(crossings[-50:], dtype=float)
        print(f"period (steps, last 50 cycles): mean={arr.mean():.1f} "
              f"std={arr.std():.1f} min={arr.min()} max={arr.max()} "
              f"n_cycles={len(crossings)}")
    else:
        print("no period detected: the oscillation is gone")

    print()
    if args.no_metabolism:
        print("control run: structure frozen (drift should be 0)")
    elif crossings and np.asarray(crossings[-50:]).std() < 0.3 * np.asarray(crossings[-50:]).mean():
        print("VERDICT: structure drifts while the oscillation stays regular. "
              "The layer ages without losing its rhythm.")
    else:
        print("VERDICT: the period is unstable, or gone. "
              "Metabolism at these rates destabilises the oscillator.")


if __name__ == "__main__":
    main()
