"""
src/L3_memory/trace.py

Connectivity structure over the state space.

What this is
------------

Input comes from the L5 state space. Each attractor records:

    "these states recur, and their centroid is at mu"

The state space records where states concentrate. It does not record
which attractors co-occur.

Memory is the second relation:

    If attractors A and B are entered consecutively, with no third
    attractor entered between them, an edge forms between A and B.

An edge is a connection weight. It grows with repeated direct transitions
and decays over time.

Why memory is not stored content
--------------------------------

In a Hopfield network memory is not located anywhere; it is a stable
configuration of the connection weights. A partial cue converges to the
nearest stable state. Nothing is retrieved: the structure is the memory.

This module therefore stores connectivity, not episodes.

Adjacency criterion
-------------------

No temporal proximity is used (explicitly rejected; see
docs/design/04_memory.md section 1.3).

The criterion is structural: a direct transition means the state moved
from one attractor to another without entering any third attractor.
Elapsed time between the two observations plays no role.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass
class Edge:
    """
    A connection between two attractors.

    The edge itself is undirected: if a transition ever happened between a and
    b, they are related, whichever direction it went. But the two directions
    are recorded and decayed separately, because they are not the same thing.
    Thinking of an apple makes you think of red far more readily than thinking
    of red makes you think of an apple; both directions exist, with different
    strengths. Nothing here enforces that they sum to a constant -- an edge can
    be strong both ways, strong one way, or weak both ways. Which it becomes is
    a product of the history, not of a rule.

    w_ab : strength of a -> b
    w_ba : strength of b -> a
    n_ab, n_ba : how many direct transitions contributed to each direction
    w    : sum of the two, for callers that only ask how related the two are
    n    : sum of the two counts

    w and n are not independent state. They are kept because several readers
    (identity, drive) only ask about the undirected relation and should not
    have to know which direction is which.
    """
    a: int
    b: int
    w_ab: float = 0.0
    w_ba: float = 0.0
    n_ab: int = 0
    n_ba: int = 0
    last_seen: int = 0          # step index of the most recent transition
    intervals: Optional[float] = None   # running mean gap between uses

    @property
    def w(self) -> float:
        return self.w_ab + self.w_ba

    @property
    def n(self) -> int:
        return self.n_ab + self.n_ba

    def touch(self, forward: bool, amount: float = 1.0, step: int = 0) -> None:
        """
        Apply one transition. Increment decreases with n; no cap.

        forward=True means the transition went a -> b.
        `step` records when, so decay can be slower for edges used rarely. The
        state space already decays regions by their own revisit interval; edges
        decaying at a uniform rate would forget rarely used paths faster than
        the regions they connect, which is the opposite of what the asymmetry
        between slow and fast structure implies.
        """
        if self.last_seen > 0 and step > self.last_seen:
            gap = float(step - self.last_seen)
            self.intervals = gap if self.intervals is None else (
                0.9 * self.intervals + 0.1 * gap
            )
        self.last_seen = int(step)
        if forward:
            self.n_ab += 1
            self.w_ab += amount / (1.0 + math.log1p(self.n_ab))
        else:
            self.n_ba += 1
            self.w_ba += amount / (1.0 + math.log1p(self.n_ba))

    def direction(self, from_node: int) -> float:
        """Strength leaving `from_node` toward the other end."""
        return self.w_ab if from_node == self.a else self.w_ba

    def other(self, node: int) -> int:
        """The node at the far end of the edge from `node`."""
        return self.b if node == self.a else self.a

    def weaken(self, factor: float) -> None:
        """Decay both directions by the same fraction."""
        self.w_ab *= max(0.0, 1.0 - factor)
        self.w_ba *= max(0.0, 1.0 - factor)


class TraceGraph:
    """
    Connectivity graph over attractor indices.

    The class does not hold centroids. Indices refer to the state space's
    attractor list, which changes as attractors are merged or removed.
    `merge_node` and `prune` keep the graph aligned.

    The graph is a downstream reader of the state space, never an editor.
    """

    def __init__(self):
        self._edges: dict[tuple[int, int], Edge] = {}
        self._last_node: Optional[int] = None

    @staticmethod
    def _key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    # ---------- record a transition ----------

    def touch(self, node: Optional[int], amount: float = 1.0,
              step: int = 0) -> Optional[Edge]:
        """
        Record that the state was assigned to attractor `node`
        (None when the state is outside every attractor).

        An edge forms only when the previous observation was also assigned
        to an attractor and it was a different one. A transition that passes
        through unassigned space does not form an edge.

        `step` is the caller's step index. It is used only to measure how
        often the edge is used, so decay can follow the same rule the state
        space uses for its regions.
        """
        if node is None:
            self._last_node = None
            return None

        prev = self._last_node
        self._last_node = node

        if prev is None or prev == node:
            return None

        k = self._key(prev, node)
        e = self._edges.get(k)
        if e is None:
            e = Edge(a=k[0], b=k[1])
            self._edges[k] = e
        e.touch(forward=(prev < node), step=step)
        return e

    # ---------- read ----------

    def neighbors(self, node: int, min_w: float = 0.0) -> list[tuple[int, float]]:
        """
        Attractors reachable from `node`, descending by the strength of the
        direction leaving `node`. The reverse direction is a different number
        and does not appear here.
        """
        out = []
        for e in self._edges.values():
            if e.a == node or e.b == node:
                dw = e.direction(node)
                if dw > min_w:
                    out.append((e.other(node), dw))
        out.sort(key=lambda t: -t[1])
        return out

    def directed(self, a: int, b: int) -> float:
        """Strength of a -> b only."""
        e = self._edges.get(self._key(a, b))
        if e is None:
            return 0.0
        return e.w_ab if a == e.a else e.w_ba

    def edge(self, a: int, b: int) -> Optional[Edge]:
        return self._edges.get(self._key(a, b))

    def strength(self, a: int, b: int) -> float:
        e = self._edges.get(self._key(a, b))
        return e.w if e is not None else 0.0

    def degree(self, node: int) -> int:
        return len(self.neighbors(node, min_w=0.0))

    def total_weight(self, node: int) -> float:
        return sum(w for _, w in self.neighbors(node))

    def __len__(self) -> int:
        return len(self._edges)

    def edges(self) -> Iterator[Edge]:
        return iter(self._edges.values())

    # ---------- decay ----------

    def weaken(self, rate: float, rng: np.random.Generator | None = None,
               spread: float = 0.0, step: int = 0,
               half_life: float = 0.0, min_scale: float = 0.01) -> None:
        """
        Reduce all edge weights.

        Not conditioned on recent use. `spread` gives each edge an
        individual rate multiplier.

        `half_life`: when positive, an edge whose own revisit interval is
        longer than this decays proportionally slower, down to `min_scale`.
        This mirrors the rule StateSpace applies to its regions. Without it,
        a rarely used path fades faster than the regions it connects, so a
        graph of sparse paths is pruned toward only the busiest routes.
        """
        if rng is None:
            rng = np.random.default_rng()
        dead = []
        for k, e in self._edges.items():
            f = rate * (1.0 + spread * float(rng.normal(0.0, 1.0)))
            if half_life > 0 and e.intervals is not None and e.intervals > 0:
                if e.intervals > half_life:
                    f *= max(half_life / e.intervals, min_scale)
            f = max(f, 0.0)
            e.weaken(f)
            if e.w <= 0.0:
                dead.append(k)
        for k in dead:
            del self._edges[k]

    # ---------- keep aligned with the state space ----------

    def merge_node(self, keep: int, drop: int) -> None:
        """
        Called when the state space merges attractor `drop` into `keep`.

        Edges of `drop` transfer to `keep`; existing parallel edges sum.
        Indices above `drop` shift down by one (the list is positional).
        """
        if keep == drop:
            return

        new_edges: dict[tuple[int, int], Edge] = {}

        def remap(i: int) -> int:
            if i == drop:
                return keep
            if i > drop:
                return i - 1
            return i

        for (a, b), e in self._edges.items():
            na, nb = remap(a), remap(b)
            if na == nb:
                continue
            k = self._key(na, nb)
            # Direction is preserved through renumbering. If the edge ran
            # a -> b before, it still runs from whichever of na, nb was a.
            w_ab, n_ab, w_ba, n_ba = e.w_ab, e.n_ab, e.w_ba, e.n_ba
            if a != k[0]:
                w_ab, n_ab, w_ba, n_ba = w_ba, n_ba, w_ab, n_ab
            if k in new_edges:
                cur = new_edges[k]
                cur.w_ab += w_ab
                cur.n_ab += n_ab
                cur.w_ba += w_ba
                cur.n_ba += n_ba
            else:
                new_edges[k] = Edge(a=k[0], b=k[1],
                                    w_ab=w_ab, n_ab=n_ab,
                                    w_ba=w_ba, n_ba=n_ba)

        self._edges = new_edges
        if self._last_node is not None:
            self._last_node = remap(self._last_node)

    def prune(self, alive_nodes: set[int]) -> None:
        """
        Drop edges touching removed attractors and renumber the remainder
        to 0..n-1 matching the state space's new list.
        """
        ordered = sorted(alive_nodes)
        remap = {old: new for new, old in enumerate(ordered)}

        new_edges: dict[tuple[int, int], Edge] = {}
        for e in self._edges.values():
            if e.a not in remap or e.b not in remap:
                continue
            na, nb = remap[e.a], remap[e.b]
            k = self._key(na, nb)
            # Reordering can swap which end is the low index; swap the
            # directional weights with it so a -> b stays a -> b.
            w_ab, n_ab, w_ba, n_ba = e.w_ab, e.n_ab, e.w_ba, e.n_ba
            if e.a != k[0]:
                w_ab, n_ab, w_ba, n_ba = w_ba, n_ba, w_ab, n_ab
            new_edges[k] = Edge(a=k[0], b=k[1],
                                w_ab=w_ab, n_ab=n_ab,
                                w_ba=w_ba, n_ba=n_ba)

        self._edges = new_edges
        if self._last_node is not None:
            self._last_node = remap.get(self._last_node)

    # ---------- observation ----------

    def stats(self) -> dict:
        if not self._edges:
            return {"edges": 0, "total_w": 0.0, "max_w": 0.0, "mean_w": 0.0,
                    "asymmetry": 0.0}
        ws = np.array([e.w for e in self._edges.values()], dtype=float)
        # Directional imbalance per edge, in [0, 1]. 0 means both directions
        # are equally strong; 1 means one direction has no weight at all.
        asym = np.array([
            abs(e.w_ab - e.w_ba) / (e.w_ab + e.w_ba)
            for e in self._edges.values() if (e.w_ab + e.w_ba) > 0
        ], dtype=float)
        return {
            "edges": len(ws),
            "total_w": float(ws.sum()),
            "max_w": float(ws.max()),
            "mean_w": float(ws.mean()),
            "asymmetry": float(asym.mean()) if asym.size else 0.0,
        }

    # ---------- persistence ----------

    def to_dict(self) -> dict:
        return {
            "edges": [
                {"a": e.a, "b": e.b,
                 "w_ab": e.w_ab, "w_ba": e.w_ba,
                 "n_ab": e.n_ab, "n_ba": e.n_ba}
                for e in self._edges.values()
            ],
            "last_node": self._last_node,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TraceGraph":
        g = cls()
        for r in d.get("edges", []):
            k = (int(r["a"]), int(r["b"]))
            if "w_ab" in r:
                g._edges[k] = Edge(a=k[0], b=k[1],
                                   w_ab=float(r["w_ab"]), n_ab=int(r["n_ab"]),
                                   w_ba=float(r["w_ba"]), n_ba=int(r["n_ba"]))
            else:
                # Older archives stored a single undirected weight. Split it
                # evenly rather than pretending the direction is known: the
                # information is simply not there.
                w = float(r.get("w", 0.0))
                n = int(r.get("n", 0))
                half_w, half_n = w / 2.0, n // 2
                g._edges[k] = Edge(a=k[0], b=k[1],
                                   w_ab=half_w, n_ab=half_n,
                                   w_ba=w - half_w, n_ba=n - half_n)
        ln = d.get("last_node")
        g._last_node = int(ln) if ln is not None else None
        return g
