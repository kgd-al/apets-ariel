#!/usr/bin/env python3
"""montage for videos: python vmontage.py out.mp4 a.mp4 b.mp4 ... [--cols 3] [--cell 640x360] [--gap 4]

Once a tile's video has finished, it keeps showing its last frame and gets a green
(success) or red (failure) marker, based on <video>.evaluation.csv.
"""

import argparse, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("videos", nargs="+", type=Path)
p.add_argument("--cols", type=int, default=2)
p.add_argument("--cell", default="500x500", help="WxH of each tile")
p.add_argument("--gap", type=int, default=0, help="pixels between tiles")
p.add_argument("--bg", default="black")
p.add_argument("--shortest", action="store_true", help="stop at the shortest input")
p.add_argument("--duration", default=None, help="maximal video duration")
p.add_argument("--mark", choices=["tint", "border"], default="tint",
               help="how a finished tile is marked (default: semi-transparent tint)")
p.add_argument("--alpha", type=float, default=0.35, help="tint opacity (0-1)")
p.add_argument("--border", type=int, default=10, help="border thickness for --mark border")
a = p.parse_args()


def success(path):
    name_parts = path.name.split(".")
    csv_path = path.with_name(name_parts[0] + ".evaluation.csv")
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, index_col=0)
    return np.isfinite(df["score"][name_parts[2]])


def duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], text=True)
    return float(out.strip()), success(path)


w, h = map(int, a.cell.split("x"))
n = len(a.videos)
rows = -(-n // a.cols)  # ceil
infos = [duration(v) for v in a.videos]

# scale each input to fit its cell (keeping aspect ratio), pad to exact cell size
chains = [
    f"[{i}:v]setpts=PTS-STARTPTS,scale={w}:{h}:force_original_aspect_ratio=decrease,"
    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:{a.bg},setsar=1[v{i}]"
    for i in range(n)
]
# row-major tile origins in absolute pixels
pos = [((i % a.cols) * (w + a.gap), (i // a.cols) * (h + a.gap)) for i in range(n)]
layout = "|".join(f"{x}_{y}" for x, y in pos)
W = a.cols * w + (a.cols - 1) * a.gap
H = rows * h + (rows - 1) * a.gap
inputs = "".join(f"[v{i}]" for i in range(n))
# a finished input holds its last frame (unless --shortest) until the longest one ends
chains.append(
    f"{inputs}xstack=inputs={n}:layout={layout}:shortest={int(a.shortest)}:fill={a.bg}[grid]"
)

# Draw the markers on the composite (not on the inputs: a finished input has no more
# timestamps), switched on from the moment each video ends.
boxes = []
for (d, ok), (x, y) in zip(infos, pos):
    if ok is None:
        continue
    color = "green" if ok else "red"
    if a.mark == "tint":
        style = f"color={color}@{a.alpha}:t=fill"
    else:
        style = f"color={color}:t={a.border}"
    boxes.append(f"drawbox=x={x}:y={y}:w={w}:h={h}:{style}:enable='gte(t,{d:.3f})'")
chains.append(f"[grid]{','.join(boxes) or 'null'}[out]")

cmd = ["ffmpeg", "-y"]
for v in a.videos:
    cmd += ["-i", v]
cmd += ["-filter_complex", ";".join(chains), "-map", "[out]", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p"]
if a.duration:
    cmd += ["-t", a.duration]
cmd += [a.out]
print(f"output: {W}x{H}")
subprocess.run(cmd, check=True)
