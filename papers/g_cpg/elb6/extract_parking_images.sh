ffmpeg -i remote/g_cpg/final/evo/locomotion/both/run-6/champion.eval.parking.mp4 -vf "fps=1/2.5" -enc_time_base 1/1000 -frame_pts 1 -fps_mode passthrough papers/g_cpg/elb6/parking/t%d.png
for f in papers/g_cpg/elb6/parking/*.png
do
    mv "$f" "papers/g_cpg/elb6/parking/$(sed \'s|.*/t\(.*\).png|\1|\' <<< $f | awk \'{printf "t%g", $1/1000}\').png"
done
