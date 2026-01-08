import time

import mujoco
from mujoco import mj_step
from mujoco.viewer import KeyCallbackType

from ..config import ViewerConfig


def passive_viewer(model, data, args: ViewerConfig):
    paused = not args.auto_start

    def callback(key: int):
        nonlocal paused
        if key == 32:
            paused = not paused

    with mujoco.viewer.launch_passive(model, data, key_callback=callback) as viewer:
        total_time = 0
        while paused:
            viewer.sync()
            time.sleep(.1)

        while viewer.is_running() and total_time < args.duration:
            step_start = time.time()

            mj_step(model, data)
            total_time += model.opt.timestep
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        if not args.auto_quit:
            paused = True

        while paused:
            time.sleep(.1)


def interactive_viewer(model, data, args: ViewerConfig):
    mujoco.viewer.launch(model, data)
