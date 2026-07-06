from ._monitor import MonitorBase
from ..mujoco.state import MjState


class WeightMonitor(MonitorBase):
    def __init__(self, robot_name: str, *args, **kwargs):
        super().__init__(frequency=None, *args, **kwargs)
        self.robot_name = robot_name

    def start(self, state: MjState):
        super().start(state)
        self._value = float(state.model.body(f"{self.robot_name}_core").subtreemass[0])
