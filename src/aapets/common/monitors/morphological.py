from ._monitor import Monitor
from ..mujoco.state import MjState


class WeightMonitor(Monitor):
    def __init__(self, name: str):
        super().__init__(frequency=None)
        self.name = name

    def start(self, state: MjState):
        self._value = float(state.model.body(f"{self.name}_core").subtreemass[0])
