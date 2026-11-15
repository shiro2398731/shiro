"""
src/L8_identity/identity.py

Identity: the connectivity that survives perturbation.

What identity is not
--------------------

* Not a self-description. Not a sentence such as "I am patient". Such a
  sentence is an L6 language product; it must derive from structure, not
  define it.

* Not a name, a trait vector, or a profile. Those are configuration and
  can be edited by anyone.

* Not a designer's statement about who the system is.

What identity is
----------------

    Identity is what remains after perturbation.

This definition requires no external standard: it asks only what persisted
through change.

Operationally: the edges whose weight rank stays in the top fraction of
the graph across repeated observations, and whose membership is stable.

The criterion is rank, not absolute weight. Absolute weight changes with
graph size; rank is relative.

Procedure
---------

At each observation the top `q` fraction of edges by weight is recorded.
An edge's presence count across observations follows

    Binomial(rounds, p = q)

if it were selected at random. An edge is retained when its count exceeds
mean + 2 standard deviations. The criterion is therefore "unlikely to be
a random selection", with the distribution supplied by the system's own
observation history.

`q` scales with graph size, so the identity set grows as the graph grows.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.L3_memory.trace import TraceGraph


@dataclass
class IdentityCore:
    """
    The set of edges constituting the current identity core.

    edges     : [(a, b, w), ...] descending by weight
    share     : mean presence rate of members across observations
    stability : same quantity, retained for compatibility
    """
    edges: list[tuple[int, int, float]]
    share: float
    stability: float

    def is_empty(self) -> bool:
        return not self.edges

    def nodes(self) -> set[int]:
        s = set()
        for a, b, _ in self.edges:
            s.add(a)
            s.add(b)
        return s

    def describe(self) -> str:
        if self.is_empty():
            return "no identity core"
        return (f"{len(self.edges)} edges, presence {self.share:.1%}")


class Identity:
    """
    Identity core.

    Read-only with respect to the graph and the state space. The only thing
    it writes is its own observation history, used as the reference
    distribution for the binomial test.
    """

    def __init__(self, top_quantile: float = 0.10, history: int = 256):
        """
        top_quantile : fraction of edges recorded per observation
        history      : number of past observations retained

        Both are observation parameters.
        """
        self.q = float(top_quantile)
        self._appear: collections.defaultdict[tuple[int, int], int] = \
            collections.defaultdict(int)
        self._rounds: int = 0
        self._history = collections.deque(maxlen=int(history))
        self._last: Optional[IdentityCore] = None
        self._edge_snapshot: dict[tuple[int, int], float] = {}

    # ---------- one observation ----------

    def observe(self, graph: TraceGraph) -> IdentityCore:
        """
        Record the top quantile of edges by weight.

        Called at the caller's observation interval.
        """
        edges = list(graph.edges())
        if not edges:
            self._rounds += 1
            self._last = IdentityCore(edges=[], share=0.0, stability=0.0)
            return self._last

        edges.sort(key=lambda e: -e.w)
        k = max(1, int(round(len(edges) * self.q)))
        top = edges[:k]

        self._rounds += 1
        for e in top:
            self._appear[(e.a, e.b)] += 1

        mean_w = float(np.mean([e.w for e in edges]))
        self._history.append(mean_w)

        stab = [self._appear[(e.a, e.b)] / self._rounds for e in top]
        stability = float(np.mean(stab)) if stab else 0.0

        total_w = sum(e.w for e in edges)
        share = float(sum(e.w for e in top) / total_w) if total_w > 0 else 0.0

        core = IdentityCore(
            edges=[(e.a, e.b, e.w) for e in top],
            share=share,
            stability=stability,
        )
        self._last = core
        return core

    # ---------- which edges constitute identity ----------

    def structural(self, min_rounds: int = 8) -> IdentityCore:
        """
        Edges whose presence count exceeds the binomial expectation.

        Binomial(rounds, p=q); retained when count > mean + 2 sd.

        min_rounds is the observation resolution: the binomial test is not
        meaningful with fewer observations. Returning empty below that
        threshold reflects insufficient data, not an absent identity.
        """
        if self._rounds < min_rounds or not self._appear:
            return IdentityCore(edges=[], share=0.0, stability=0.0)

        p = self.q
        n = self._rounds
        mu = n * p
        sd = float(np.sqrt(n * p * (1.0 - p)))
        cutoff = mu + 2.0 * sd

        chosen = [(a, b) for (a, b), c in self._appear.items() if c > cutoff]
        if not chosen:
            return IdentityCore(edges=[], share=0.0, stability=0.0)

        stab = [self._appear[(a, b)] / n for a, b in chosen]
        mean_stab = float(np.mean(stab))
        edges = [(a, b, float(self._edge_snapshot.get((a, b), 0.0)))
                 for a, b in chosen]
        edges.sort(key=lambda r: -r[2])
        return IdentityCore(
            edges=edges,
            share=mean_stab,
            stability=mean_stab,
        )

    def snapshot(self, graph: TraceGraph) -> None:
        """Record current edge weights for use by structural()."""
        self._edge_snapshot = {(e.a, e.b): e.w for e in graph.edges()}

    # ---------- fingerprint ----------

    def fingerprint(self, state_space, graph: TraceGraph) -> dict:
        """
        A small set of quantities describing the identity core, for
        comparison across carriers.

        n_nodes       : number of attractors in the core
        n_edges       : number of edges in the core
        mean_w        : mean weight in the core
        centroid_norm : norm of the centroid of the core's attractors
        span          : median pairwise distance among them
        """
        self.snapshot(graph)
        core = self.structural()
        if core.is_empty():
            return {"n_nodes": 0, "n_edges": 0, "mean_w": 0.0,
                    "centroid_norm": 0.0, "span": 0.0}

        nodes = core.nodes()
        attrs = state_space.attractors()
        row = state_space.uid_row()
        # The core names regions by uid; the geometry lives in rows. The order
        # here exists only so the float sums below have one canonical
        # evaluation order -- it is the centroid bytes, not the name, so no
        # ranking anywhere depends on what a region is called.
        nodes = [u for u in nodes if u in row]
        nodes.sort(key=lambda u: attrs[row[u]].mu.tobytes())

        mus = np.stack([attrs[row[u]].mu for u in nodes]) if nodes else None
        centroid_norm = float(np.linalg.norm(mus.mean(axis=0))) if mus is not None else 0.0

        span = 0.0
        if len(nodes) >= 2:
            d = []
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    d.append(float(np.linalg.norm(mus[i] - mus[j])))
            span = float(np.median(d))

        return {
            "n_nodes": len(nodes),
            "n_edges": len(core.edges),
            "mean_w": float(np.mean([w for _, _, w in core.edges])),
            "centroid_norm": centroid_norm,
            "span": span,
        }

    # ---------- observation ----------

    @property
    def rounds(self) -> int:
        return self._rounds

    @property
    def last(self) -> Optional[IdentityCore]:
        return self._last
