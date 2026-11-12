"""
src/L5_cognition/stream.py

Public interface of L5.

Two levels of output:

  1. Continuous quantities (`stream_state`) -- consumed by L3, L4, L8.
     Includes the state vector.

  2. Discrete symbols (`SymbolBundle`) -- consumed by L6.
     Contains attractor index, sequence, intent. No vectors.

L6 cannot obtain any vector: the interface does not provide one
(ARCHITECTURE.md section 4.2 whitelist). This is enforced by what is
exposed, not by convention.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .dynamics import DynamicState, Dynamics
from .landscape import Landscape
from .terrain import StateSpace
from .transitions import Transition, TransitionDetector
from .workspace import GlobalWorkspace, Ignition
from .adaptive_gain import ErrorDrivenGain
from .grounding import GroundingTable
from .self_model import OnlineSelfModel
from src.L3_memory import Consolidator, Recall, RecallEngine, TraceGraph
from src.L4_drive import Drive
from src.L4_drive.regulation import Regulation
from src.L5_cognition.rest import RestField
from src.L5_cognition.swing import Swing
from src.L8_identity import Identity


@dataclass
class SymbolBundle:
    """
    The only symbols L5 exposes to L6.

    attractor_id is the index of the nearest salient attractor, or "none".
    It carries no semantic content.

    referent is a discrete label for what that region has most often meant in
    her own terms (see grounding.GroundingTable): a code for where two measured
    quantities sat inside her own history. Still a label, not a vector, because
    §4.2 forbids L6 from receiving vectors.
    """
    attractor_id: str
    transition_seq: list[str]
    intent: str
    t_event: str
    referent: str = "none"


class Stream:
    """Public interface of L5."""

    def __init__(
        self,
        dim: int = 128,
        gamma: float = 0.05,
        dt: float = 0.01,
        recover_rate: float = 0.002,
        max_attractors: int = 8192,
        seed: int = 0,
    ):
        self.D = int(dim)
        self.landscape = Landscape(
            dim=dim, recover_rate=recover_rate, max_attractors=max_attractors
        )
        self.dynamics = Dynamics(dim=dim, gamma=gamma, dt=dt)
        self.detector = TransitionDetector(self.landscape.state_space)
        self.workspace = GlobalWorkspace()
        self.rng = np.random.default_rng(seed)

        self._st: Optional[DynamicState] = None

        # Recovery interval, counted in integration steps rather than time.
        # This is an observation/engineering interval, not a rate of the
        # system itself.
        self._recover_every = 16
        # Let the state space know the cadence, so recovery can relate a rate
        # to a number of steps.
        self.landscape.state_space._recover_every_hint = self._recover_every
        self._step_count = 0
        self._last_ignition: Optional[Ignition] = None

        # ---- L3 memory: connectivity structure over the state space ----
        self.graph = TraceGraph()
        self.recaller = RecallEngine()
        self.consolidator = Consolidator()
        self._last_recall: Optional[Recall] = None

        # ---- L4 drive: unresolved tension ----
        self.drive = Drive(dim=dim)

        # ---- L4 regulation: viability-driven pull ----
        # Attached here rather than inside StateSpace so the state space stays
        # a pure potential field; the modulation is a separate layer.
        self.regulation = Regulation()
        # The traversability edge is off unless explicitly enabled. See step().
        self._wall_enabled = False

        # ---- L8 identity: the subset that survives perturbation ----
        self.identity = Identity()
        self._identity_every = 64
        self._last_core = None

        # ---- M1: her model of her own next state, and the error it makes ----
        # The model is first order over the regions she occupies; the gain it
        # drives is bounded and only scales exploration. The prediction itself
        # is never fed back into the dynamics -- if it were, her state would be
        # following her expectation of it instead of the potential.
        #
        # Both live on the stream (not in the runtime loop) so that they are
        # part of her state: the L5 payload carries them, and a restart resumes
        # the same self-model instead of a blank one.
        self.self_model = OnlineSelfModel()
        self.self_gain = ErrorDrivenGain()
        self._last_symbol: Optional[int] = None

        # ---- L6 grounding: what a region has meant, in her own terms ----
        # Lives here, not in L6, because it needs continuous quantities (speed,
        # tension) that L6 is not allowed to receive. L6 gets only the label.
        self.grounding = GroundingTable()

        # Rest and swing were built while trying to find an internal source of
        # motion: without noise the state settled and stopped, and two attempts
        # to keep it moving were a per-region slow variable and a global slow
        # direction. Neither worked. The thing that did work was the prediction
        # error, because the oscillator is always wrong about its own next step.
        #
        # Once that was in place these two were left in by default, and an
        # ablation says they are not earning their place:
        #
        #     rest  swing   att  edges   speed  trans
        #      on     on    526    285   0.347    459
        #      on    off    551    307   0.348    406
        #     off     on    432    196   0.442    436
        #     off    off    479    241   0.413    356
        #
        # No consistent direction, differences within the run-to-run spread.
        # Module kept in the tree, since the argument for removing it is a
        # measurement rather than a preference, but not attached by default.
        self._use_rest = False
        self._use_swing = False
        self.rest = RestField()
        self._rest_every = 1
        self.swing = Swing(dim=dim)

        # L4's tilt is set from the structure's own gradient scale rather than
        # from a unit length. Sampled once the first step has happened, since
        # before that there is no state to measure at.
        self._drive_gain_set = False
        self._drive_grad_samples: list[float] = []

    @property
    def state_space(self) -> StateSpace:
        return self.landscape.state_space

    # Compatibility alias.
    @property
    def terrain(self) -> StateSpace:
        return self.landscape.state_space

    # ---------- starting point ----------

    def calibrate(self, l0_samples: np.ndarray) -> None:
        """
        Set the zero point from warm-up L0 statistics and initialise the state.

        This is the only source of where the system starts.
        """
        zero = self.landscape.calibrate_zero(l0_samples)
        self._st = self.dynamics.init_state(zero)

    @property
    def started(self) -> bool:
        return self._st is not None

    # ---------- advance ----------

    def step(self) -> Optional[Transition]:
        """
        Advance one integration step.

        Call frequency is decided by the caller (the staleness criterion),
        not by this method.

        After advancing, the workspace is observed once. Ignition events are
        recorded in workspace.history and do not affect the state space or
        the trajectory.
        """
        if self._st is None:
            raise RuntimeError("Stream not calibrated: no zero point, no state")

        self._st = self.dynamics.step(self._st, self.landscape, self.rng)
        self._step_count += 1

        # Register the observed state.
        a = self.landscape.record(
            self._st.x, datetime.now(timezone.utc).isoformat()
        )

        # Swing advances from the position the state is in now, before the
        # next step uses its tilt. Off by default; see __init__.
        if self._use_swing:
            self.swing.observe_gradient(
                float(np.linalg.norm(self.landscape.gradient(self._st.x)))
            )
            self.swing.step(self._st.x)
            self.landscape.set_swing(self.swing.tilt())

        # Give the drive the same scale as the structure's own pull. Sampled
        # over the first steps rather than once: a single gradient magnitude
        # depends on where the state happens to be, and one sample makes the
        # tilt strength a property of that moment rather than of the structure.
        if not self._drive_gain_set:
            _g = float(np.linalg.norm(self.landscape.gradient(self._st.x)))
            if _g > 0:
                self._drive_grad_samples.append(_g)
            if len(self._drive_grad_samples) >= 32:
                self.drive.set_baseline(list(self._drive_grad_samples))
                self._drive_gain_set = True

        # Rest advances every step, so a region the state sits in loses its
        # hold gradually. Off by default; see __init__.
        attrs_now = self.landscape.state_space.attractors()
        if self._use_rest:
            occupied = None
            if a is not None:
                for i, item in enumerate(attrs_now):
                    if item is a:
                        occupied = i
                        break
            for i in range(len(attrs_now)):
                self.rest.touch(i)
            self.rest.step(occupied)
            self.landscape.state_space.set_rest(
                [self.rest.value(i) for i in range(len(attrs_now))]
            )

        # L3: connect consecutive attractors when the transition is direct.
        node: Optional[int] = None
        if a is not None:
            attrs = self.landscape.state_space.attractors()
            for i, item in enumerate(attrs):
                if item is a:
                    node = i
                    break
            self.graph.touch(node, step=self._step_count)

            # M1: score her prediction of this region against the region she
            # actually reached, then learn the transition. The prediction was
            # made from the previous symbol, before this one existed.
            #
            # A state the model has never seen produces no prediction, and no
            # prediction produces no error: counting that as a miss would make
            # the gain read "exploration" as "being wrong about myself".
            prev, self._last_symbol = self._last_symbol, node
            if prev is not None:
                predicted = self.self_model.predict(prev)
                if predicted is not None:
                    self.self_gain.observe(predicted == node)
                self.self_model.observe(prev, node)

        # L3: which attractors the current state co-activates.
        # Read-only: it does not modify the state space or the graph.
        self._last_recall = self.recaller.by_cue(
            self.landscape.state_space, self.graph, self._st.x
        )

        # L4: recompute the tension field for the next step.
        #
        # Placed after the graph has been updated. It used to run before
        # record() and graph.touch(), so every step's tensions were computed
        # from the previous step's connectivity and then applied on the step
        # after that: a two-step lag between what the structure was and what
        # the drive said about it. The tilt cannot use the state that has not
        # happened yet, but the graph it reads is already up to date here.
        self.landscape.set_drive(
            self.drive.deflection(self.landscape.state_space, self.graph)
        )

        # L6 grounding: record which region this is *and* where two measured
        # quantities sat inside her own history at the same moment. Done at the
        # end of the step, on purpose: the symbol has to be the one L6 will
        # receive (`detector.current_attractor`), not the index in the attractor
        # list. Measured with the list index instead, the engine's table scored
        # MI = 1.0240 bits while a probe keyed on the broadcast id scored 0.7395
        # -- two different numberings of the same regions.
        #
        # Both buckets are None until her history is long enough, and an
        # unmeasured context is not counted as "low".
        _speed = Dynamics.speed(self._st)
        _tension = float(self.drive.summary().get("sum", 0.0))
        b_speed = self.grounding.bucket("speed", _speed)
        b_tension = self.grounding.bucket("tension", _tension)

        # Recovery, counted in integration steps.
        if self._step_count % self._recover_every == 0:
            self.landscape.recover()
            # The half-life the state space uses for its own regions, so edges
            # and regions are forgotten under the same rule.
            self.consolidator.metabolize(
                self.graph, rng=self.rng, step=self._step_count,
                half_life=float(getattr(self.landscape.state_space, "half_life", 0.0)),
            )

            # Regulation is recalculated at the same cadence as recovery:
            # viability is defined by the gain/loss balance, and the loss side
            # only changes when recovery is applied.
            ss = self.landscape.state_space
            self.landscape.state_space.set_pull(self.regulation.factors(ss))
            # The traversability edge follows the same measurement: where the
            # structure can no longer hold a state is derived from the depths
            # it currently maintains.
            #
            # Disabled by default. Measured: with the edge active, the fraction
            # of regions below their own carrying level fell from 0.94 to 0.60,
            # but the distance the state covered grew from 6 to 150 region
            # scales. The edge pushes the state away from thin regions instead
            # of letting it settle, which is the opposite of what resilience
            # means (absorbing disturbance, not deflecting it).
            if self._wall_enabled:
                self.landscape.state_space.set_wall(
                    self.regulation.viability.wall(ss))
            else:
                self.landscape.state_space.set_wall(None)

        # Attractor count may have changed through recovery or merging.
        self.consolidator.sync(self.graph, self.landscape.state_space)

        # Workspace: read-only.
        self._last_ignition = self.workspace.observe(
            self.landscape.state_space, self._st.x,
            datetime.now(timezone.utc).isoformat()
        )

        # L8: record structure. The interval is an observation resolution.
        if self._step_count % self._identity_every == 0:
            self.identity.snapshot(self.graph)
            self._last_core = self.identity.observe(self.graph)

        ev = self.detector.observe(
            self._st, datetime.now(timezone.utc).isoformat()
        )

        # L6 grounding, second half: the symbol is the id the detector just
        # published, and the context was measured above, in the same step.
        aid = self.detector.current_attractor
        if aid is not None and b_speed is not None and b_tension is not None:
            self.grounding.observe(int(aid), (b_speed, b_tension))

        return ev

    # ---------- continuous output ----------

    def broadcast_state(self) -> dict:
        """
        The fields the broadcast actually carries, and nothing else.

        stream_state() computes salient counts, a regulation summary (which
        re-runs the viability measurement over every region) and a drive
        summary. Those are for observation and for a snapshot taken every few
        minutes. The broadcast is sent on every transition, and downwind only
        attractor_id is read: L6 is allowed the id and no continuous quantity.
        Building the full dictionary on every transition did that work for
        nothing.
        """
        st = self._st
        return {
            "attractor_id": (self.detector.current_attractor
                             if self.detector.current_attractor is not None else -1),
            "potential": (self.landscape.potential(st.x) if st is not None else 0.0),
            # Kept because StreamState carries a vector and the contract says so.
            # It is not read by L6; the whitelist in the adapter is what enforces
            # that, not the absence of the field here.
            "x": st.x.tolist() if st is not None else [],
        }

    def stream_state(self) -> dict:
        st = self._st
        ss = self.landscape.state_space
        return {
            "x": st.x.tolist() if st is not None else [],
            "speed": Dynamics.speed(st) if st is not None else 0.0,
            "attractor_id": (self.detector.current_attractor
                             if self.detector.current_attractor is not None else -1),
            "potential": (self.landscape.potential(st.x) if st is not None else 0.0),
            "n_attractors": len(ss),
            "n_salient": len(ss.salient()),
            "global_scale": ss.global_scale,
            "viability": self.regulation.summary(ss),            # L3
            "n_edges": len(self.graph),
            "recall": (list(self._last_recall.path)
                       if self._last_recall is not None else []),
            # L4
            "tensions": self.drive.summary(),
            # L8
            "identity": (len(self._last_core.edges)
                         if self._last_core is not None else 0),
        }

    # Compatibility alias for the previous field names.
    def stream_state_legacy(self) -> dict:
        d = self.stream_state()
        return {
            "x": d["x"],
            "speed": d["speed"],
            "attractor_id": d["attractor_id"],
            "potential": d["potential"],
            "n_traces": d["n_attractors"],
            "n_basins": d["n_salient"],
        }

    # ---------- discrete output ----------

    def get_symbols(self) -> SymbolBundle:
        """The only thing L6 receives. No vectors, no continuous quantities."""
        aid = self.detector.current_attractor
        seq = self.detector.recent_sequence(k=8)
        return SymbolBundle(
            attractor_id=str(aid) if aid is not None else "none",
            transition_seq=[str(s) for s in seq],
            intent="state" if aid is not None else "silent",
            t_event=datetime.now(timezone.utc).isoformat(),
            # A label, not a number: the region's most usual position in her
            # own measured context, or "none" when that is not yet measurable.
            referent=(self.grounding.label(int(aid)) if aid is not None else "none"),
        )
