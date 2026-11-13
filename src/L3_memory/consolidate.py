"""
src/L3_memory/consolidate.py

Maintenance of the connectivity graph.

Two operations only:

  1. sync      -- keep edge indices aligned with the state space's
                  attractor list (merging and removal change it)
  2. metabolize -- reduce edge weights

No third operation
------------------

An earlier draft included a step that would "strengthen" high-weight
edges. That is not implemented: an edge's importance is already its
weight. Adding a second multiplier requires an external criterion for
what counts as important, and this system does not have one.

On replay
---------

Some systems compress experience by replaying it. That serves to reclaim
context budget, which this system does not have. Compression here is
already produced by merging: when two attractors describe the same
region, StateSpace merges them and the graph follows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .trace import TraceGraph


class Consolidator:
    """
    Maintains the graph's alignment with the state space.

    Produces nothing; does not schedule itself.
    `sync` is called after the state space changes, `metabolize` after
    recovery. The call frequency is decided by the caller.
    """

    def __init__(self, weaken_rate: float = 0.002, spread: float = 0.5):
        """
        weaken_rate : base edge decay rate
        spread      : per-edge individual variation

        Both are rates of the medium, comparable to StateSpace.recover_rate.
        """
        self.weaken_rate = float(weaken_rate)
        self.spread = float(spread)

    # ---------- alignment ----------

    def sync(self, graph: TraceGraph, state_space,
             remap: Optional[dict] = None) -> dict:
        """
        Align the graph with the current attractor list.

        `remap` is the state space's old->new index map for the structural
        change that just happened (`StateSpace.take_index_remap`). It is the
        only correct input: the old version assumed the list only ever loses
        its *last* entries and kept edges with index below the new length, so
        removing a middle region left every later edge pointing at its
        neighbour. Passing the map makes `prune` (which renumbers survivors in
        order) line the graph up exactly.

        Without a map the old behaviour is kept for callers that only need a
        bounds check.
        """
        n = len(state_space)
        before = len(graph)
        if remap is not None:
            if all(e.a < n and e.b < n for e in graph.edges()) and len(remap) == n:
                return {"dropped_edges": 0}
            graph.prune({int(k) for k in remap.keys()})
            return {"dropped_edges": before - len(graph)}
        alive = {i for i in range(n)}
        # If nothing has changed, rebuilding the whole edge dictionary every
        # step is pure cost. Rewriting it is O(E) with a Python loop over every
        # edge; at a few hundred edges that is a meaningful share of a 3 ms step.
        if all(e.a < n and e.b < n for e in graph.edges()) and not getattr(graph, "_needs_renumber", False):
            return {"dropped_edges": 0}
        graph.prune(alive)
        return {"dropped_edges": before - len(graph)}

    # ---------- decay ----------

    def metabolize(self, graph: TraceGraph,
                   rng: Optional[np.random.Generator] = None,
                   step: int = 0,
                   half_life: float = 0.0) -> None:
        """
        Reduce edge weights.

        Slower than state-space recovery, because connectivity is a
        first-order relation between attractors and outlasts their depth.

        `half_life`: when positive, edges used rarely decay proportionally
        slower, the same rule StateSpace applies to regions.
        """
        graph.weaken(self.weaken_rate, rng=rng, spread=self.spread,
                     step=step, half_life=half_life)

    # ---------- observation ----------

    def report(self, graph: TraceGraph, state_space) -> dict:
        """Current graph statistics. Observation only."""
        n = len(state_space)
        st = graph.stats()
        degrees = [graph.degree(i) for i in range(n)] if n else []
        return {
            "nodes": n,
            "edges": st["edges"],
            "total_w": round(st["total_w"], 4),
            "max_w": round(st["max_w"], 4),
            "mean_degree": round(float(np.mean(degrees)), 3) if degrees else 0.0,
            "max_degree": int(max(degrees)) if degrees else 0,
        }
