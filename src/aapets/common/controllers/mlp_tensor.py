import numpy as np
import torch
from torch import nn

from .abstract import Controller
from ..mujoco.state import MjState


def mlp_structure(hinges: int, width: int, depth: int) -> nn.Sequential:
    sizes = [hinges] + [width] * depth + [hinges]
    seq = nn.Sequential()
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])

        seq.append(linear)
        seq.append(nn.Tanh())

    return seq


def mlp_weights(sequence: nn.Sequential, weights: np.ndarray):
    _wi, _wn = 0, 0
    for m in sequence.children():
        if isinstance(m, nn.Linear):
            _sn = m.weight.data.size()
            _wn = int(np.prod(_sn))
            m.weight.data[:] = torch.from_numpy(weights[_wi:_wi + _wn].reshape(_sn))
            _wi += _wn

            _wn = int(np.prod(m.bias.data.size()))
            m.bias.data[:] = torch.from_numpy(weights[_wi:_wi + _wn])
            _wi += _wn

    assert _wi == len(weights), f"Error: {len(weights)-_wi} unused weights"


class MLPTensorBrain(Controller):
    def __init__(
            self,
            weights: np.ndarray,
            state: MjState,
            name: str,
            depth: int, width: int):

        super().__init__(weights, state, name)

        self._modules = mlp_structure(len(self._joints_pos), width, depth)
        mlp_weights(self._modules, weights)

    @classmethod
    def name(cls): return "mlp_tensor"

    @classmethod
    def num_parameters(cls, state: MjState, name: str, width: int, depth: int, *args, **kwargs):
        return sum(
            p.numel() for p in
            mlp_structure(cls.num_joints(state, name), width, depth).parameters()
        )

    def extract_weights(self) -> np.ndarray:
        params = []
        for param in self._modules.parameters():
            params.append(param.view(-1))
        return np.array(params)

    def __call__(self, state: MjState) -> None:
        state = [a.qpos for a in self._joints_pos.keys()]
        action = self._modules(torch.tensor(state)).detach().numpy()

        for i, (actuator, ctrl) in enumerate(zip(self._actuators, action)):
            actuator.ctrl[:] = ctrl * self._ranges[i]
