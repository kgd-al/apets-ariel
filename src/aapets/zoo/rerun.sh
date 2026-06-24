#!/bin/bash

venv=ariel-venv

tasks=.tasks

find $@ -name "summary.csv" | sort | while read f
do
  echo $(dirname $f)/champion.zip
done > $tasks

nstasks=$(cat $tasks | wc -l)
echo "$ntasks tasks"

worker(){
  source ${PYENV_ROOT}/versions/$venv/bin/activate

  i=0
  cat .tasks | while read archive
  do
  #  printf "\n\033[32m[%6.2f%%]\033[0m " \$(( 100 * \$i / $ntasks ))
    awk -vi=$i -vn=$ntasks 'BEGIN{printf "\n\033[32m[%6.2f%%]\033[0m ", 100 * i / n}'
    i=$(($i+1))
    echo $archive
    python -m aapets.bin.rerun --robot-archive $archive \
      --viewer NONE --movie \
       --camera "apet1_tracking-cam" --camera-distance 2 --camera-angle 45 --camera-center com \
      --plot-format png --plot-trajectory --plot-brain-activity
  done

  printf "\n\033[32m[100.00%%] Done\033[0m\n"

  rm $tasks
}

srun --nodes=1 --ntasks 1 --cpus-per-task 1 --job-name=rerun-all --time=10:00:00 worker

