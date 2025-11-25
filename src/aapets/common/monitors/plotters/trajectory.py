from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from .._monitor import Monitor
from ...mujoco.state import MjState


class TrajectoryPlotter(Monitor):
    def __init__(self, frequency, name: str):
        super().__init__(frequency)
        self.name = name
        self.data, self.core = None, None

    def start(self, state: MjState):
        self.core = state.data.body(f"{self.name}_core").xpos
        self.data = [[] for _ in range(3)]

    def _step(self, state: MjState):
        for i, v in enumerate([state.time, *self.core[:2]]):
            self.data[i].append(v)

    def plot(self, path: Path):
        df = pd.DataFrame({k: d for k, d in zip(["time", "x", "y"], self.data)})
        df[["x", "y"]] -= df[["x", "y"]].iloc[0, :]

        v_max = df[["x", "y"]].abs().max().max()

        fig, ax = plt.subplots()

        if v_max > 0:
            ax.set_xlim(-v_max, v_max)
            ax.set_ylim(-v_max, v_max)
        ax.set_aspect("equal")
        ax.set_title(f"Trajectory in {round(df.time.iloc[-1]):g}s")

        ax.plot(df["x"], df["y"])

        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
