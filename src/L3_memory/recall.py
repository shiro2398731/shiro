"""
src/L3_memory/recall.py

Pattern completion over the connectivity structure.

Two paths
---------

**1. Cued completion**

Hopfield-style pattern completion, extended to a continuous space.

In a Hopfield network a partial cue converges to the nearest stable state.
Nothing is retrieved; the structure is the memory.

Here the attractors are the stable states. Completion is:

    1. the current state x falls within one characteristic scale of an
       attractor centroid -> that attractor is activated
    2. activation propagates along the graph from that attractor

Step 2 matters: a single activated attractor states only that one
attractor dominates here. Activation spreading along edges is the
operation in which several attractors become active together.

**2. Un-cued activation**

No cue is used; activation starts from the graph itself.

In neuroscience this corresponds to sharp-wave ripples: previously
active patterns replay without an external cue.

The starting edge is the strongest in the graph. No judgement is made
about "what is worth thinking about"; the connection weight determines
where activity starts.

What this module does not do
----------------------------

It writes nothing. It does not modify the state space or the graph.

docs/design/04_memory.md section 1.2 states that recall changes structure.
That change is produced by the trajectory: activation does not move the
state. The state moves only through Dynamics, and any deepening is the
result of a state actually being observed there.

Timing
------

No time window is applied. The reach of activation is bounded by the edge
weights themselves, via max_depth and decay below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .trace import TraceGraph


@dataclass
class Recall:
    """
    One completion result.

    path    : attractor indices in activation order. path[0] is the one the
              cue matched directly; the rest were reached through edges.
    weights : same length as path. weights[0] == 1.0 (direct match, no edge).
    spontaneous : True when no cue was used.
    """
    path: list[int]
    weights: list[float]
    spontaneous: bool = False

    def __len__(self) -> int:
        return len(self.path)

    def is_empty(self) -> bool:
        return not self.path

    def describe(self) -> str:
        if self.is_empty():
            return "no activation"
        head = "spontaneous" if self.spontaneous else "cued"
        return f"{head}, {len(self.path)} attractors: " + \
               " > ".join(str(i) for i in self.path)


class RecallEngine:
    """
    Pattern completion over a TraceGraph.

    Reads the state space (to find the matched attractor) and the graph
    (to propagate). Writes nothing.
    """

    def __init__(
        self,
        max_depth: int = 4,
        decay: float = 0.5,
    ):
        """
        max_depth : maximum number of edges traversed.
        decay     : activation retained per edge.

        Both bound the computation. They do not define what counts as
        activation; that is determined by the edge weights.
        """
        self.max_depth = int(max_depth)
        # Kept for the floor only: propagation still stops after max_depth hops,
        # and the floor is expressed relative to that bound. The per-step
        # attenuation itself comes from the edge weights.
        self.decay = float(decay)

    # ---------- cued ----------

    def by_cue(
        self,
        state_space,
        graph: TraceGraph,
        x: np.ndarray,
    ) -> Recall:
        """
        The current state is the cue.

        Step 1: find the attractor whose normalized distance to x is below 1.
        Step 2: propagate from it along the graph.
        """
        x = np.asarray(x, dtype=float)
        attrs = state_space.attractors()
        if not attrs:
            return Recall(path=[], weights=[])

        # Vectorised lookup. This runs once per step, and a Python loop over
        # np.linalg.norm was taking half the step time. The centroid and scale
        # matrices come from the state space's own cache rather than being
        # rebuilt from the attractor list on every step.
        try:
            state_space._centroid_matrix()
            mus = state_space._mu_cache
            sig = np.maximum(state_space._sigma_vec, np.finfo(float).eps)
        except AttributeError:
            mus = np.stack([a.mu for a in attrs])
            sig = np.maximum(
                np.array([a.sigma for a in attrs], dtype=float),
                np.finfo(float).eps,
            )
        d = mus - x
        z = np.sqrt(np.einsum("ij,ij->i", d, d)) / sig
        best_i = int(np.argmin(z))
        best_z = float(z[best_i])

        if best_z >= 1.0:
            # No attractor within one characteristic scale of the state.
            return Recall(path=[], weights=[])

        path, weights = self._spread(graph, best_i, first_weight=1.0)
        return Recall(path=path, weights=weights, spontaneous=False)

    # ---------- un-cued ----------

    def spontaneous(
        self,
        graph: TraceGraph,
        rng: np.random.Generator | None = None,
    ) -> Recall:
        """
        Start from the strongest edge in the graph.

        Selection criterion: the edge with the largest weight. Noise (rng)
        is used only to break ties, never to select randomly among
        weakly-ordered candidates.

        The cutoff is the median weight plus one median absolute deviation:
        an edge must exceed the graph's typical weight to start activation.
        """
        if len(graph) == 0:
            return Recall(path=[], weights=[], spontaneous=True)

        ws = np.array([e.w for e in graph.edges()], dtype=float)
        if ws.size == 0:
            return Recall(path=[], weights=[], spontaneous=True)

        med = float(np.median(ws))
        mad = float(np.median(np.abs(ws - med)))
        if mad < 1e-12:
            mad = max(abs(med), float(np.sqrt(np.finfo(float).eps)))
        cutoff = med + 1.0 * mad

        # A spontaneous thought starts somewhere and travels one way. The edge
        # qualifies if either direction is strong; which direction it then
        # takes is decided in _spread, by the directional weight.
        candidates = [(e.a, e.b, e.w) for e in graph.edges() if e.w >= cutoff]
        if not candidates:
            return Recall(path=[], weights=[], spontaneous=True)

        if rng is not None and len(candidates) > 1:
            pick = candidates[int(rng.integers(0, len(candidates)))]
        else:
            pick = max(candidates, key=lambda t: t[2])

        a, b, _ = pick
        # Enter the edge from whichever end has the stronger outgoing weight;
        # an edge that is only strong one way starts from the strong side.
        if graph.directed(a, b) >= graph.directed(b, a):
            start, second = a, b
        else:
            start, second = b, a
        path, weights = self._spread(graph, start, first_weight=1.0, second=second)
        return Recall(path=path, weights=weights, spontaneous=True)

    # ---------- propagation ----------

    def _spread(
        self,
        graph: TraceGraph,
        start: int,
        first_weight: float = 1.0,
        second: Optional[int] = None,
    ) -> tuple[list[int], list[float]]:
        """
        Propagate from `start` along the strongest available edge at each hop.

        Not all neighbours activate; only the strongest one continues. A
        lower bound of first_weight * decay^max_depth terminates propagation.
        Visited nodes are not revisited.
        """
        path = [start]
        weights = [first_weight]
        visited = {start}

        cur = start
        cur_w = first_weight

        if second is not None and second not in visited:
            path.append(second)
            weights.append(cur_w * self.decay)
            visited.add(second)
            cur = second
            cur_w = weights[-1]

        floor = first_weight * (self.decay ** self.max_depth)

        while len(path) <= self.max_depth:
            nbrs = graph.neighbors(cur)
            fresh = [(n, w) for n, w in nbrs if n not in visited]
            if not fresh:
                break
            nxt, edge_w = fresh[0]
            # The strength of the step is the edge's share of everything
            # leaving this node, not a fixed fraction.
            #
            # Every hop used to be multiplied by the same constant, so a
            # connection that has been walked a hundred times carried activation
            # exactly as far as one walked twice. The weights were computed and
            # then discarded: the loop bound `nxt, _w = nbrs[0]` never used `_w`.
            #
            # A share is in [0, 1] and needs no extra scale, which is what a
            # fixed decay factor was standing in for. When there is only one
            # outgoing edge its share is 1 and the activation carries without
            # loss, which is the right answer: there was nothing to choose
            # between.
            #
            # Called once and reused: neighbours was being computed twice per
            # hop, which showed up as a step time going from 1.7 to 3.3 ms.
            total_out = 0.0
            for _, w in nbrs:
                total_out += w
            share = (edge_w / total_out) if total_out > 0 else 0.0
            nw = cur_w * share
            if nw < floor:
                break
            path.append(nxt)
            weights.append(nw)
            visited.add(nxt)
            cur, cur_w = nxt, nw

        return path, weights
