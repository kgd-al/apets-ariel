from typing import Tuple, List

from .base import GenericFetchDynamics
from aapets.fetch.controllers.fetcher import FetcherCPG
from ..overlay import FetchOverlay
from ..types import InteractionMode
from ...common.monitors.plotters.brain_activity import BrainActivityPlotter
from ...common.monitors.plotters.trajectory import TrajectoryPlotter
from ...common.mujoco.state import MjState


class DemoDynamics(GenericFetchDynamics):
    def __init__(self,
                 state: MjState,
                 overlay: FetchOverlay,
                 robot: str, ball: str, human: str,
                 brain: FetcherCPG):

        super().__init__(
            state, InteractionMode.DEMO, overlay,
            robot, ball, human, brain
        )

        self.script: List[Tuple[float, float]] = [
            (0, 1), (0, -1),
            (-1, 1), (+1, 1),
            (0, 0), (+1, -1), (-1, -1), (0, 0)
        ]
        self._switch_period = self.duration() / len(self.script)
        self._next_switch, self._current_stage = 0, -1

    @classmethod
    def duration(cls): return 60

    def _step(self, state: MjState):
        if self._next_switch < state.data.time:
            self._current_stage += 1
            self._next_switch = self._next_switch + self._switch_period

            alpha, beta = self.script[self._current_stage]
            self.brain.overwrite_modulators(alpha, beta)

        super()._step(state)

    def postprocess(self, brain_monitor: BrainActivityPlotter, trajectory_monitor: TrajectoryPlotter):
        path = brain_monitor.path
        fig = brain_monitor.plot(None)

        lines = [i * self._switch_period for i in range(len(self.script))]
        ticks = [l + .5 * self._switch_period for l in lines]
        ticks_labels = [f"α={a}; β={b}" for a, b in self.script]

        for ax in fig.axes:
            ax.autoscale(axis='x', tight=True)
            ax.tick_params(axis='x', which='minor', length=0, pad=20)

            for l in lines:
                ax.axvline(l, color='r')
                ax.set_xticks(ticks, ticks_labels, minor=True)

        fig.savefig(path, bbox_inches='tight')

    def _process_keys(self): pass
