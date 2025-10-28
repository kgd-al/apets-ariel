from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mujoco import MjModel, MjData

from aapets.config import CommonConfig


class ControlAndTrack:
    def __init__(self, brain, tracked_objects: dict[str, Any],
                 config: CommonConfig):

        self.config = config

        self._control_step = 1 / config.control_frequency
        self._next_control_step = 0

        self.brain = brain
        self._plot_brain_activity = config.plot_brain_activity
        if self._plot_brain_activity:
            self._brain_activity_data = [[] for _ in range(2 * self.brain.cpgs + 1)]

        sample = False

        self.tracked_objects = tracked_objects
        self._track_trajectory = config.plot_trajectory
        if config.plot_trajectory:
            sample = True
            self._trajectory_data = [[] for _ in range(3)]
            self._robot_core = self.tracked_objects[f"{config.robot_name_prefix}1_core"]

        if sample:
            self._sample_step = 1 / config.sampling_frequency
            self._next_sample_step = 0
        else:
            self._next_sample_step = np.nan

    def mjcb_callback(self, model: MjModel, data: MjData):
        if data.time >= self._next_control_step:
            self.brain.control(model, data)

            if self.plot_brain_activity:
                self.track_brain_activity(data)

            self._next_control_step += self._control_step

        if data.time >= self._next_sample_step:
            if self._track_trajectory:
                self.track_trajectory(data)

            self._next_sample_step += self._sample_step

    def track_brain_activity(self, mj_data: MjData):
        data = self._brain_activity_data
        data[0].append(mj_data.time)
        for i, (act, r) in enumerate(zip(self.brain.actuators, self.brain.ranges)):
            data[2 * i + 1].append(act.length / r)
            data[2 * i + 2].append(act.ctrl / r)

    def plot_brain_activity(self, path: Path):
        if not self._plot_brain_activity:
            raise RuntimeError("Brain activity was not monitored and, thus, cannot be plotted.")

        w, h = matplotlib.rcParams["figure.figsize"]
        n = self.brain.cpgs

        fig, axes = plt.subplots(n, 2,
                                 sharex=True, sharey=True,
                                 figsize=(3 * w, 2 * h))

        data = self._brain_activity_data
        x = np.array(data[0])
        for i in range(n):
            for j, label in enumerate(["Position", "Control"]):
                ax = axes[i][j]
                ix = 2 * i + j + 1

                ax.plot(x, data[ix], zorder=1)
                title = self.brain.actuators[i].name
                if i == 0:
                    title = f"{label}\n\n" + title

                ax.set_title(title)

        fig.tight_layout()
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")

    def track_trajectory(self, mj_data: MjData):
        data = self._trajectory_data
        core_pos = self._robot_core.xpos

        for i, v in enumerate([mj_data.time, *core_pos[:2]]):
            data[i].append(v)

    def plot_trajectory(self, path: Path):
        df = pd.DataFrame({k: d for k, d in zip(["time", "x", "y"], self._trajectory_data)})
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
