ffmpeg -i remote/g_cpg/final/evo/locomotion/both/run-6/champion.eval.+figure8.mp4 -vf "fps=1/5,crop=iw:ih/2" -enc_time_base 1 -frame_pts 1 -fps_mode passthrough papers/g_cpg/elb6/figure8/t%d.png
