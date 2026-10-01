#!/usr/bin/env python3
"""montage for videos: python vmontage.py out.mp4 a.mp4 b.mp4 ... [--cols 3] [--cell 640x360] [--gap 4]"""
import argparse, subprocess

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("videos", nargs="+")
p.add_argument("--cols", type=int, default=2)
p.add_argument("--cell", default="500x500", help="WxH of each tile")
p.add_argument("--gap", type=int, default=0, help="pixels between tiles")
p.add_argument("--bg", default="black")
p.add_argument("--shortest", action="store_true", help="stop at the shortest input")
a = p.parse_args()

w, h = map(int, a.cell.split("x"))
n = len(a.videos)
rows = -(-n // a.cols)  # ceil

# scale each input to fit its cell (keeping aspect ratio), pad to exact cell size
chains = [
    f"[{i}:v]setpts=PTS-STARTPTS,scale={w}:{h}:force_original_aspect_ratio=decrease,"
    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:{a.bg},setsar=1[v{i}]"
    for i in range(n)
]
# row-major layout in absolute pixels
layout = "|".join(
    f"{(i % a.cols) * (w + a.gap)}_{(i // a.cols) * (h + a.gap)}" for i in range(n)
)
W = a.cols * w + (a.cols - 1) * a.gap
H = rows * h + (rows - 1) * a.gap
inputs = "".join(f"[v{i}]" for i in range(n))
chains.append(
    f"{inputs}xstack=inputs={n}:layout={layout}:shortest={int(a.shortest)}:fill={a.bg}[out]"
)

cmd = ["ffmpeg", "-y"]
for v in a.videos:
    cmd += ["-i", v]
cmd += ["-filter_complex", ";".join(chains), "-map", "[out]", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", a.out]
print(f"output: {W}x{H}")
subprocess.run(cmd, check=True)
