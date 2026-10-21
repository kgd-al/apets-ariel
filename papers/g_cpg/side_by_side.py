#!/usr/bin/env python3
"""Two videos side by side, each with a caption.

python side_by_side.py out.mp4 left.mp4 right.mp4 "Left label" "Right label" [--gap 20]

Both inputs are assumed to share size / fps / format. The video is re-encoded once
(unavoidable: stacking and drawing text change the pixels); audio, if kept, is copied.
"""
import argparse, subprocess, tempfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("left")
p.add_argument("right")
p.add_argument("label_left")
p.add_argument("label_right")
p.add_argument("--gap", type=int, default=0, help="pixels between the videos")
p.add_argument("--bg", default="black", help="gap colour")
p.add_argument("--fontsize", type=int, default=32)
p.add_argument("--font", default=None, help="path to a .ttf (default: fontconfig's default)")
p.add_argument("--position", choices=["bottom", "top"], default="bottom")
p.add_argument("--audio", choices=["none", "left", "right"], default="none")
p.add_argument("--crf", type=int, default=18, help="x264 quality (lower = better)")
p.add_argument("--preset", default="fast", help="x264 preset")
a = p.parse_args()

with tempfile.TemporaryDirectory() as tmp:
    # captions go through text files so no drawtext/filtergraph escaping is needed
    files = []
    for name, text in (("l", a.label_left), ("r", a.label_right)):
        f = Path(tmp) / f"{name}.txt"
        f.write_text(text, encoding="utf-8")
        files.append(f)

    y = "h-th-20" if a.position == "bottom" else "20"
    font = f":fontfile={a.font}" if a.font else ""

    def label(f):
        return (f"drawtext=textfile={f}:expansion=none:x=(w-text_w)/2:y={y}"
                f":fontsize={a.fontsize}:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=8{font}")

    chains = [f"[0:v]{label(files[0])}[l0]", f"[1:v]{label(files[1])}[r]"]
    # the gap is padding on the right side of the left video
    chains.append(f"[l0]pad=iw+{a.gap}:ih:0:0:{a.bg}[l]" if a.gap else "[l0]null[l]")
    chains.append("[l][r]hstack=inputs=2[out]")

    cmd = ["ffmpeg", "-y", "-i", a.left, "-i", a.right,
           "-filter_complex", ";".join(chains), "-map", "[out]"]
    if a.audio != "none":
        cmd += ["-map", f"{0 if a.audio == 'left' else 1}:a?", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-preset", a.preset, "-crf", str(a.crf),
            "-pix_fmt", "yuv420p", a.out]
    subprocess.run(cmd, check=True)
