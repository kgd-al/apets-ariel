import numpy as np
import torch
from torch import nn

from .abstract import Controller
from ..mujoco.state import MjState


def mlp_structure(*, hinges: int, width: int, depth: int, grad: bool) -> nn.Sequential:
    sizes = [hinges] + [width] * depth + [hinges]
    seq = nn.Sequential()
    for i in range(depth + 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        linear.requires_grad_(grad)

        seq.append(linear)
        if i < depth:
            seq.append(nn.Tanh())

    return seq


def mlp_weights(sequence: nn.Sequential, weights: np.ndarray):
    _wi, _wn = 0, 0
    weights = weights.astype(np.float32)
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
            depth: int, width: int, grad: bool = False):

        # print("[kgd-debug|MLPTensor:__init__]")
        super().__init__(weights, state, name)

        self._modules = mlp_structure(hinges=len(self._joints_pos), width=width, depth=depth, grad=grad)
        mlp_weights(self._modules, weights)

    @classmethod
    def name(cls): return "mlp_tensor"

    @classmethod
    def num_parameters(cls, state: MjState, name: str, width: int, depth: int, *args, **kwargs):
        return cls.num_parameters_from_module(
            mlp_structure(hinges=cls.num_joints(state, name), width=width, depth=depth, grad=False)
        )

    @classmethod
    def num_parameters_from_module(cls, module: nn.Module):
        return sum(p.numel() for p in module.parameters())

    def extract_weights(self) -> np.ndarray:
        params = []
        for param in self._modules.parameters():
            params.append(param.view(-1))
        return np.array(params)

    @staticmethod
    def observation(joints, ranges, state):
        # print("[kgd-debug|MLPTensor] >> observation =",
        #       {j.name: f"{j.qpos[0]} / {r}" for j, r in zip(joints, ranges)})
        return torch.as_tensor(np.array([j.qpos[0] / r for j, r in zip(joints, ranges)],
                                        dtype=np.float32))
        # Actuator length may not be reliable / consistent. Joints seem to be
        return torch.as_tensor(np.array([a.length[0] / r for a, r in zip(actuators, ranges)],
                                        dtype=np.float32))

    def __call__(self, state: MjState) -> None:
        observation = self.observation(self._joints, self._ranges, state)
        actions = np.clip(self._modules(observation).cpu().numpy(), -1, 1)

        for i, (actuator, ctrl) in enumerate(zip(self._actuators, actions)):
            actuator.ctrl[:] = ctrl * self._ranges[i]

        # print(f"[kgd-debug|MLPTensor:__call__] t={state.time}")
        # print(f"[kgd-debug|MLPTensor:__call__] qpos={state.data.qpos}")
        # print(f"[kgd-debug|MLPTensor:__call__] {observation=}")
        # print(f"[kgd-debug|MLPTensor:__call__] {actions=}")
        #
        # with np.printoptions(precision=50):
        #     print(f"[kgd-debug|MLPTensor:__call__] ctrl={state.data.ctrl}")
