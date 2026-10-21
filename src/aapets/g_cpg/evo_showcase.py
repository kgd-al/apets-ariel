
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Annotated

import numpy as np
import pandas as pd
from rich.progress import track

from aapets.bin.rerun import Arguments as RerunArguments, main as rerun
from aapets.common.misc.config_base import IntrospectiveAbstractConfig
from aapets.common.robot_storage import RerunnableRobot

from aapets.g_cpg.config import Config

@dataclass
class Arguments(IntrospectiveAbstractConfig):
    folder: Annotated[Path, "Root folder of the evolution, containing best*.zip files",
                      dict(required=True)] = None

    duration: Annotated[float, "Total video duration"] = 15
    threads: Annotated[int, "Number of parallel threads"] = 1

    brain_first_gen: Annotated[int, "Generation after which to show the brains"] = 0


def make_movie(archive: Path, speed: float):
    rerun_args = RerunArguments()
    rerun_args.robot_archive = archive
    
    rerun_args.movie = "mp4"
    rerun_args.camera = "apet1_tracking-cam"
    rerun_args.camera_angle = 45
    rerun_args.camera_distance = 2
    rerun_args.camera_center = "com"

    rerun_args.movie_speed = speed
    rerun_args.movie_speed_overlay = False

    rerun_args.plot_trajectory = True

    rerun_args.verbosity = -1

    rerun_args.render_brain_phenotype = True
    rerun_args.plot_format = "png"

    rerun(rerun_args)

    return pd.read_csv(archive.with_suffix(".trajectory.csv"))["x"].iloc[-1]


def video_width(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL,
    ).stdout
    return int(out.strip().split(",")[0])

def annotate_video(src, dst, text, image=None, frac=0.25, margin=20,
                   fontfile=None, fontsize=28):
    """Text top-left, optional image top-right at `frac` of the video width."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write(text)
    try:
        opts = [f"textfile={tf.name}", f"x={margin}", f"y={margin}", f"fontsize={fontsize}",
                "fontcolor=white", "box=1", "boxcolor=black@0.5", "boxborderw=10"]
        if fontfile:
            opts.insert(0, f"fontfile={fontfile}")
        drawtext = "drawtext=" + ":".join(opts)

        cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(src)]
        if image:
            pic_w = int(video_width(src) * frac) // 2 * 2          # even width
            cmd += ["-i", str(image)]
            fc = (f"[0:v]{drawtext}[txt];"
                  f"[1:v]scale={pic_w}:-1[pic];"
                  f"[txt][pic]overlay=W-w-{margin}:{margin}[out]")
        else:
            fc = f"[0:v]{drawtext}[out]"
        cmd += ["-filter_complex", fc, "-map", "[out]", "-map", "0:a?",
                "-c:a", "copy", str(dst)]

        subprocess.run(cmd, stdin=subprocess.DEVNULL,
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed:\n{e.stderr}") from e
    finally:
        Path(tf.name).unlink()


def annotate(movie, speed, generation):
    out = Path(movie).with_name(Path(movie).stem + "_annotated.mp4")
    brain = None
    if generation >= args.brain_first_gen:
        brain = movie.with_suffix(".sym_abcpg.png")

    annotate_video(movie, out, f"gen {generation}\nspeed {speed:.3f} m/s", image=brain)
    return out


def merge_movies(movies, output):
    movies = [Path(m).resolve() for m in movies]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lst:
        for m in movies:
            escaped = str(m).replace("'", r"'\''")   # escape single quotes for the list format
            lst.write(f"file '{escaped}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0",
             "-i", lst.name, "-c", "copy", output],
            stdin=subprocess.DEVNULL,
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed:\n{e.stderr}") from e
    finally:
        Path(lst.name).unlink()


def sorted_movies(files):
    return list(sorted(files, key=lambda s: int(s.stem.split("_")[1])))
        
def main():
    base_duration = None
    last_brain, last_specs = None, None
    candidates = []
    for f in sorted_movies(args.folder.glob("best_*.zip")):
        record = RerunnableRobot.load(f)
        if len(candidates) == 0:
            is_new = True
            base_duration = record.config.duration

        else:
            is_new = (
                not np.array_equal(last_brain, record.brain[2])
            ) or (
                last_specs != record.mj_spec.to_xml()
            )

        if is_new:
            last_brain = record.brain[2]
            last_specs = record.mj_spec.to_xml()
            candidates.append(f)

    movie_speed = len(candidates) * base_duration / args.duration

    ncpus = args.threads or len(os.sched_getaffinity(0) - 1)
    results = {}
    with ProcessPoolExecutor(ncpus) as pool:
        futures = {pool.submit(make_movie, f, movie_speed): f for f in candidates}
        for fut in track(as_completed(futures), total=len(futures), description="Making movies"):
            results[futures[fut]] = fut.result()

    with ProcessPoolExecutor(ncpus) as pool:
        def movie_of(f): return f.with_suffix(".mp4")
        def gen_of(f): return int(f.stem.split("_")[1])
        futs = [pool.submit(annotate, movie_of(f), speed, gen_of(f)) for f, speed in results.items()]
        annotated = [fu.result() for fu in futs]   # input order preserved

    output = args.folder.joinpath("evolution.mp4")
    merge_movies(sorted_movies(annotated), output)

    print(f"Merged {len(candidates)} into", output)

if __name__ == "__main__":
    args = Arguments.parse_command_line_arguments("Small tool to make merge evolution videos")
    main()
