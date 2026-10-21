step=5
x=250
y=150
w=1750
h=600

rm papers/g_cpg/hardware/robodog-slalom/t*.png
ffmpeg -i papers/g_cpg/hardware/richard-slalom_2.truncated.mp4 -vf "fps=1/$step,crop=$w:$h:$x:$y" -enc_time_base 1 -frame_pts 1 -fps_mode passthrough papers/g_cpg/hardware/robodog-slalom/t%d.png
