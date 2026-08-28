from datetime import timedelta
import os
import shutil
import time

import humanize
import numpy as np
import pandas as pd

from aapets.common.monitors._monitor import MonitorBase
os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass
from pathlib import Path

from mujoco import mj_forward, mj_step, mjtGeom, mju_euler2Quat, MjSpec

from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress

from aapets.common import controllers
from aapets.common.metrics_storage import BAD, GOOD, RESET, EvaluationMetrics
from aapets.common.monitors import metrics
from aapets.common.monitors.plotters.record import MovieRecorder
from aapets.common.mujoco.callback import MjcbCallbacks
from aapets.common.mujoco.state import MjState
from aapets.common.robot_storage import RerunnableRobot
from aapets.bin.rerun import Arguments as RerunArguments
from aapets.zoo.config import Arguments as ZooArguments
from aapets.common.world_builder import adjust_side_camera


@dataclass
class Arguments(RerunArguments):
    pass


class SmoothShifter(MonitorBase):
    def __init__(self, duration, final_y, target_y, *args, **kwargs):
        super().__init__(frequency=1000)
        self.duration = duration
        self.target_y = target_y
        self.body, self.qpos_adr = None, None

    def start(self, state: MjState):
        super().start(state)
        self.body = state.data.body("apet1_world")
        jnt_id = state.model.body(self.body.name).jntadr[0]
        self.qpos_adr = state.model.jnt_qposadr[jnt_id]

    def _step(self, state: MjState):
        super()._step(state)
        y = self.target_y * state.time / self.duration
        state.data.qpos[self.qpos_adr + 1] = y

    def stop(self, state: MjState):
        super().stop(state)


def simulate_one(record: RerunnableRobot, file: Path, args: Arguments,
                 final_y: float = None, target_y: float = None):
    state, model, data = MjState.from_spec(record.mj_spec).unpacked
    mj_forward(model, data)

    brain = controllers.get(record.brain[0])(
        weights=record.brain[2], state=state, name=args.robot_name_prefix, **record.brain[1])

    monitors = dict(movie_recorder=MovieRecorder(
        args.movie_framerate, args.movie_width, args.movie_height,
        file,
        camera=args.camera, shadows=True
    ))
    if final_y is not None:
        monitors["shifter"] = SmoothShifter(args.duration, final_y, target_y)

    with MjcbCallbacks(state, [brain], monitors, args) as callback:
        mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

    return callback.metrics


def add_good(spec: MjSpec, dx, dy):  # Coin
    coin_body = spec.worldbody.add_body(name="coin", pos=[dx, dy, .08])
    coin_body.add_geom(
        name="coin_geom",
        type=mjtGeom.mjGEOM_CYLINDER,
        euler=[np.pi / 2, 0, 0],  # tips it onto its edge
        size=[0.08, 0.01, 0],       # radius, half-thickness -> thin disc
        rgba=[1.0, 0.85, 0.1, 1],  # gold
    )


def add_bad(spec: MjSpec, dx, dy):  # Black hole
    spec.delete(spec.geom("floor"))
    
    half_extent = 10.0      # matches your old plane's x/y half-size
    depth = 100
    nrow, ncol = 256, 256   # resolution — higher = smoother, more memory

    data = np.ones((nrow, ncol), dtype=np.float32)  # 1.0 = flat, max height everywhere

    # convert a world-space (x, y) into a (row, col) index into the grid
    def world_to_grid(x, y):
        row = int((y + half_extent) / (2 * half_extent) * (nrow - 1))
        col = int((x + half_extent) / (2 * half_extent) * (ncol - 1))
        return row, col

    pit_row, pit_col = world_to_grid(dx, -dy)
    pit_radius_cells = 4

    yy, xx = np.mgrid[0:nrow, 0:ncol]
    dist = np.sqrt((yy - pit_row) ** 2 + (xx - pit_col) ** 2)
    mask = dist < pit_radius_cells
    data[mask] = 0

    spec.add_hfield(
        name="ground_hfield",
        nrow=nrow, ncol=ncol,
        size=[half_extent, half_extent, depth, 1],  # x, y half-extent, elevation, base
        userdata=data.flatten(),
    )

    spec.worldbody.add_geom(
        name="ground",
        type=mjtGeom.mjGEOM_HFIELD,
        hfieldname="ground_hfield",
        pos=[0, 0, -depth],
        material="floor",  # reuse your existing checker material if you had one
    )


def work(archive: Path, args: Arguments):
    record = RerunnableRobot.load(archive)
    output_prefix = archive.with_suffix("")

    if args.is_default("duration"):
        args.duration = record.config.duration

    #### Default simulation, just with bigger ground

    args.camera = f"{args.robot_name_prefix}1_tracking-cam"
    args.camera_angle = 45
    args.camera_distance = 2
    adjust_side_camera(record.mj_spec, args, args.robot_name_prefix)

    spec = record.mj_spec
    floor = spec.geom("floor")  # or spec.geom("floor") depending on your mujoco version
    floor.size = [10, 10, 0.1]  # 20x20 units, grid spacing 0.1

    movie_file = output_prefix.with_suffix(".mp4")
    simulate_one(record, movie_file, args)

    print(f"Generated {movie_file}")

    #### Tweaked simulation, with positive and negative outcomes

    final_x, final_y = pd.read_csv(archive.with_suffix(".trajectory.csv"))[["x", "y"]].iloc[-1]

    time_ratio = args.duration / record.config.duration
    if time_ratio != 1:
        final_x *= time_ratio
        final_y *= time_ratio

    dx, dy = 1.1 * final_x, .5

    add_good(spec, dx, dy)
    add_bad(spec, dx, dy)

    args.camera = "pretty-cam"
    camera = spec.camera(args.camera)
    camera.pos = [.5 * final_x, -1.25 * final_x, 1.25 * final_x]
    mju_euler2Quat(camera.quat, [.25 * np.pi, 0, 0], "xyz")

    movie_file = output_prefix.with_suffix(".good.mp4")
    simulate_one(record, movie_file, args, final_y=final_y, target_y=dy)
    print(f"Generated {movie_file}")

    movie_file = output_prefix.with_suffix(".bad.mp4")
    simulate_one(record, movie_file, args, final_y=final_y, target_y=-dy)
    print(f"Generated {movie_file}")

    return archive
    
def main(args: Arguments):
    start = time.perf_counter()
    
    archives = sorted(list(Path("remote/zoo/__champions/").glob("*/champion.zip")))
    n = len(archives)

    args.duration = 10

    with Progress() as progress, ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        task = progress.add_task("Processing...", total=n)
        futures = [executor.submit(work, archive, args) for archive in archives]
        failures = 0
        for future in as_completed(futures):
            archive = future.result()
            progress.update(task, advance=1, description=archive)

    output_folder = Path("remote/zoo/__hs_videos")
    output_folder.mkdir(exist_ok=True, parents=True)
    for a in archives:
        for v in ["", ".good", ".bad"]:
            src = a.with_suffix(f"{v}.mp4")
            dst = output_folder.joinpath(f"{a.parent.name}{v}.mp4")
            print(src, "->", dst)
            shutil.copyfile(src, dst)

    if failures == 0:
        msg = f"{GOOD}Successfully processed {n} items{RESET}"
    else:
        msg = f"{BAD}Failed to process {failures} archives (out of {n}){RESET}"

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    print(msg, f"in {duration}s")


if __name__ == "__main__":
    main(Arguments.parse_command_line_arguments(
        description="Simple utility to generate all videos for the human study"))
