step=2
x=800
y=600
w=400
h=200

ffmpeg -ss 1:08 -to 1:26 -i papers/g_cpg/hardware/football_1.MP4 -vf "fps=1/$step,crop=$w:$h:$x:$y" -enc_time_base 1 -frame_pts 1 -fps_mode passthrough papers/g_cpg/hardware/football/t%d.png
