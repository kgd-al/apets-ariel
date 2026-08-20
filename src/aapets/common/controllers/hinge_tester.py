import numpy as np

from ..mujoco.state import MjState
from .abstract import Controller


class HingeTesterBrain(Controller):
    @classmethod
    def name(cls): return "hinge-tester"

    def __init__(self, state: MjState, name: str):
        super().__init__([], state, name)
        self.single_hinge_duration = 2 # seconds
        self.duration = self.hinges * self.single_hinge_duration

    def __call__(self, state: MjState):
        n, t = self.hinges, state.time
        angles = [0] * n
        angle = np.sin(t * 2 * np.pi / self.single_hinge_duration)
        i = min(int(state.time // self.single_hinge_duration), n-1)
        angles[i] = angle

        for i, (actuator, ctrl) in enumerate(zip(self._actuators, angles)):
            actuator.ctrl[:] = ctrl * self._ranges[i]

    def extract_weights(self): return np.array()
    def set_weights(self, weights): pass

    @classmethod
    def num_parameters(self): return 0

    def reset(self, state: MjState):
        pass