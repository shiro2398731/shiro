"""
scripts/_test_p2_loop.py

Does P2 run as a loop with its own clock and its own inputs, not just as a
script calling stream.step()?

Every measurement so far has driven L5 directly from a test script. P2 is the
process that actually does it in the running system: it waits on the bus, gates
L2 updates, feeds the state space, publishes on transition, and persists on a
wall-clock interval. None of that has been exercised.

This starts P2 wiring in the same order it starts in the process, publishes a
synthetic upstream signal, and checks that events come out the other side.

Run yourself:
    .\\venv\\Scripts\\python.exe scripts\\_test_p2_loop.py
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import src.common.persist as persist
from src.L5_cognition.stream import Stream
from src.L5_cognition.noise import ErrorNoise
from src.runtime.process_1_fast import FHNNetwork, L1Predictor


class Upstream:
    def __init__(self, seed=42):
        self.net = FHNNetwork(seed=seed)
        self.pred = L1Predictor(input_dim=100)
        self.noise = ErrorNoise(dim=128)

    def step(self):
        sn = self.net.step()
        self.pred.step(sn["E_rate_vec"])
        return self.noise.observe(self.pred.last_error, self.pred.proj)


def main():
    s = Stream(dim=128)
    loaded = persist.load_l5(s)
    print(f"archive loaded: {loaded}")
    try:
        s.step()
    except RuntimeError:
        s.calibrate(np.random.default_rng(1).normal(0.5, 0.1, (64, 128)))
        print("calibrated from a synthetic upstream")

    up = Upstream()

    # Reproduce P2's loop body without the bus: the point here is that the
    # pieces are driven in the same order, not that ZeroMQ works.
    transitions = 0
    broadcasts = 0
    persisted = 0
    last_persist = time.time()
    PERSIST_INTERVAL_S = 1.0      # shortened so the test exercises it

    t0 = time.time()
    for i in range(4000):
        s.landscape.set_noise_amplitude(up.step())
        ev = s.step()
        if ev is not None:
            transitions += 1
            _ = s.broadcast_state()
            broadcasts += 1
        if time.time() - last_persist >= PERSIST_INTERVAL_S:
            persist.save_l5(s)
            persisted += 1
            last_persist = time.time()

    el = time.time() - t0
    ss = s.stream_state()
    print()
    print(f"steps        : {i + 1} in {el:.1f}s ({el / (i + 1) * 1000:.2f} ms/step)")
    print(f"attractors   : {ss['n_attractors']}")
    print(f"edges        : {ss['n_edges']}")
    print(f"transitions  : {transitions}")
    print(f"broadcasts   : {broadcasts}")
    print(f"persist saves: {persisted}")
    print(f"speed        : {ss['speed']:.4f}")

    ok_moves = ss["speed"] > 0
    ok_trans = transitions > 0
    ok_persist = persisted > 0
    ok_edges = ss["n_edges"] > 0

    print()
    print("=" * 55)
    checks = [
        ("state moves", ok_moves),
        ("transitions happen", ok_trans),
        ("edges form", ok_edges),
        ("persistence fires", ok_persist),
    ]
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")
    if all(ok for _, ok in checks):
        print()
        print("P2's loop body runs without a bus and produces state, "
              "transitions, edges and checkpoints.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
