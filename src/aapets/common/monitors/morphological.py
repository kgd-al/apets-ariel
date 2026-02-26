from ._monitor import Monitor
from ..mujoco.state import MjState


class WeightMonitor(Monitor):
    def __init__(self, robot_name: str):
        super().__init__(frequency=None)
        self.robot_name = robot_name

    def start(self, state: MjState):
        self._value = float(state.model.body(f"{self.robot_name}_core").subtreemass[0])
