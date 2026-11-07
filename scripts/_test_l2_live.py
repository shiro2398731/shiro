"""
scripts/_test_l2_live.py

Does L2 actually receive cycle statistics in the real loop?

Dimensions 5, 6 and 7 of the state field are built from cycle-level statistics
supplied by the envelope detector. In an earlier isolated test they were
constant at 0.500 -- the normalisation's value for "no data" -- because the test
passed cycle_stats=None. That test therefore said nothing about whether the real
loop supplies them.

This runs the same three objects P1 runs, in the same order, with the real
envelope detector, and reports per-dimension variance. A dimension that stays at
0.500 here is dead in production too.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_l2_live.py
"""

import collections
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.runtime.process_1_fast import (
    FHNNetwork, L1Predictor, L2StateField, EnvelopeDetector, StalenessGate,
)


def main():
    net = FHNNetwork(seed=42)
    pred = L1Predictor(input_dim=100, proj_dim=8)
    field = L2StateField()
    env = EnvelopeDetector()
    gate = StalenessGate(threshold=1.0)

    s_t_hist = collections.deque(maxlen=100)
    vectors = []
    cycle_seen = 0
    l2_updates = 0
    last_fed = None

    t0 = time.time()
    for i in range(20000):
        snap = net.step()
        s_t = pred.step(snap["E_rate_vec"])
        s_t_hist.append(s_t)

        now = time.time()
        phase, period = env.update(snap["fast"], now)
        cs = env.last_cycle_stats

        # The gate decides whether L2 recomputes, from the same six inputs the
        # real loop feeds it. Outside the gate the field does not advance,
        # which is the point of the staleness criterion.
        gate_inputs = {
            "s_t": s_t,
            "fast": snap["fast"],
            "I": snap["I_rate"],
            "mid": snap["mid"],
            "slow": snap["slow"],
            "pressure": snap["pressure"],
        }
        if gate.should_update(gate_inputs) or not vectors:
            v, dev = field.step(
                s_t=s_t, fast=snap["fast"], I=snap["I_rate"],
                mid=snap["mid"], slow=snap["slow"],
                pressure=snap["pressure"], cycle_stats=cs,
            )
            gate.commit(gate_inputs)
            l2_updates += 1
            if i > 4000:
                vectors.append(np.asarray(v, dtype=float))
            if cs is not None and cs is not last_fed:
                field.feed_cycle(cs)
                last_fed = cs
        if cs is not None and cs is not last_fed:
            field.feed_cycle(cs)
            last_fed = cs
            cycle_seen += 1

    el = time.time() - t0
    V = np.stack(vectors) if vectors else np.zeros((0, 16))

    print(f"{i+1} steps in {el:.1f}s, L2 recomputed {l2_updates} times "
          f"({l2_updates/(i+1):.1%}), cycles seen {cycle_seen}")
    if V.size == 0:
        print("no L2 samples at all")
        return

    print()
    print(f"{'dim':>4} {'std':>10} {'min':>8} {'max':>8}  status")
    dead = []
    for d in range(16):
        sd = float(V[:, d].std())
        lo = float(V[:, d].min())
        hi = float(V[:, d].max())
        st = "DEAD" if sd < 1e-6 else "ok"
        if sd < 1e-6:
            dead.append(d)
        print(f"{d:>4} {sd:>10.6f} {lo:>8.3f} {hi:>8.3f}  {st}")

    print()
    print(f"dead dimensions: {dead if dead else 'none'}")
    if 5 in dead or 6 in dead or 7 in dead:
        print("cycle-based dimensions are dead in the real loop: "
              "the cycle statistics are not reaching the field.")
    elif dead:
        print("some dimensions are dead; they are not the cycle-based ones.")
    else:
        print("every dimension carries data in the real loop.")


if __name__ == "__main__":
    main()
