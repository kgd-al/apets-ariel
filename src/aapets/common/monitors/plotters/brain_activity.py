from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from .._monitor import Monitor
from ...mujoco.state import MjState


class BrainActivityPlotter(Monitor):
    """Monitors brain activity by storing joints I/O

    A bit primitive but works fine.
    .. warning:: cannot discriminate between robots
    """

    def __init__(self, frequency, name, path: Path):
        super().__init__(frequency)
        self.name, self.path = name, path
        self.data, self.actuators = [], None

    def start(self, state: MjState):
        joints = [
            j.name for j in state.spec.worldbody.find_all("joint")
            if self.name in j.name
        ]
        self.actuators = {j: state.data.actuator(j) for j in joints}
        self.data = [[] for _ in range(2 * len(self.actuators) + 1)]

    def stop(self, state: MjState):
        self.plot(self.path)

    def _step(self, state: MjState):
        self.data[0].append(state.time)
        for i, act in enumerate(self.actuators.values()):
            self.data[2 * i + 1].append(act.length)
            self.data[2 * i + 2].append(act.ctrl)

    def plot(self, path: Path):
        w, h = matplotlib.rcParams["figure.figsize"]
        n = len(self.actuators)

        fig, axes = plt.subplots(n, 2,
                                 sharex=True, sharey=True,
                                 figsize=(3 * w, 2 * h))

        x = np.array(self.data[0])
        for i, name in enumerate(self.actuators.keys()):
            for j, label in enumerate(["Position", "Control"]):
                ax = axes[i][j]
                ix = 2 * i + j + 1

                ax.plot(x, self.data[ix], zorder=1)
                title = name
                if i == 0:
                    title = f"{label}\n\n" + title

                ax.set_title(title)

        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
