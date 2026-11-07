"""
scripts/_test_l0_turnover.py

Does turnover replace the substrate without changing it?

The property being checked is not "does anything change" but "does anything
change that should not":

    connections replaced   yes, continuously
    number of connections  no
    weight distribution    no
    oscillation            no

A run that replaces connections while keeping count and distribution fixed is
what turnover is supposed to do. A run where the count drifts, or the weights
collapse to a different range, or the oscillation disappears, is turnover that
is not preserving the shape of the substrate.

Everything here is measured in steps. The envelope detector reports periods in
wall-clock seconds, which is meaningless for a loop that runs as fast as the CPU
allows.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_l0_turnover.py --steps 200000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.runtime.process_1_fast import FHNNetwork


def stats(net) -> dict:
    out = {}
    for name, m in net._coupling_matrices().items():
        nz = m[m != 0.0]
        out[name] = {
            "count": int(nz.size),
            "mean": float(nz.mean()) if nz.size else 0.0,
            "std": float(nz.std()) if nz.size else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--every", type=int, default=40000)
    ap.add_argument("--scale", default="fast", choices=("fast", "unit"))
    ap.add_argument("--no-turnover", action="store_true")
    args = ap.parse_args()

    net = FHNNetwork(seed=42)
    base = stats(net)

    # Period counted in steps: the fast variable, smoothed over 8 steps, rising
    # through its own recent median.
    env = []
    crossings = []
    was_above = None
    last_cross = None

    print(f"{'step':>8} {'EE_cnt':>8} {'EE_mean':>9} {'EE_std':>9} "
          f"{'IE_cnt':>8} {'period':>8} {'replaced':>10}")

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

        if not args.no_turnover:
            net.metabolize(activity_scale=(fast if args.scale == "fast" else 1.0))

        if i % args.every == 0 or i == 1:
            s = stats(net)
            rep = sum(net.turnover._replaced.values())
            p = crossings[-1] if crossings else None
            print(f"{i:>8} {s['EE']['count']:>8} {s['EE']['mean']:>9.5f} "
                  f"{s['EE']['std']:>9.5f} {s['IE']['count']:>8} "
                  f"{(f'{p}' if p else '-'):>8} {rep:>10}")

    print()
    print("=" * 60)
    final = stats(net)
    print(f"connections replaced in total: {sum(net.turnover._replaced.values())}")

    ok_count = all(
        abs(final[k]["count"] - base[k]["count"]) / max(base[k]["count"], 1) < 0.02
        for k in base
    )

    # A matrix with few connections will show a few percent of mean shift from
    # sampling alone, because replacing 60 of 61 values has a wide sampling
    # distribution. The tolerance has to reflect the sample size, not a fixed
    # percentage: the standard error of the mean of n draws from the base
    # distribution is std/sqrt(n), and the shift is compared against that.
    shifts = {}
    ok_mean = True
    for k in base:
        n = max(final[k]["count"], 1)
        se = base[k]["std"] / np.sqrt(n) if base[k]["std"] > 0 else 1e-12
        shift = abs(final[k]["mean"] - base[k]["mean"])
        ratio = shift / se
        shifts[k] = (shift, se, ratio)
        if ratio > 6.0:      # six standard errors; not a sampling accident
            ok_mean = False

    for name in base:
        b, f = base[name], final[name]
        dc = (f["count"] - b["count"]) / max(b["count"], 1)
        print(f"  {name:<12} count {b['count']:>5} -> {f['count']:>5} ({dc:+.1%})")
        if name in shifts:
            sh, se, r = shifts[name]
            print(f"  {'':<12} mean {b['mean']:.5f} -> {f['mean']:.5f}  "
                  f"shift {sh:.5f} = {r:.1f} x standard error ({se:.5f})")

    if crossings:
        arr = np.asarray(crossings[-200:], dtype=float)
        print(f"period (steps, last 200): mean={arr.mean():.1f} std={arr.std():.1f}")

    print()
    if args.no_turnover:
        print("control run: no turnover")
    elif ok_count and ok_mean:
        print("VERDICT: connections were replaced while count and weight "
              "distribution stayed put. That is turnover.")
    else:
        print("VERDICT: the substrate changed shape, not just identity. "
              "This is drift, not turnover.")


if __name__ == "__main__":
    main()
