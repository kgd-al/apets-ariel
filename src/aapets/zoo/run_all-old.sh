#!/bin/bash

usage(){
    echo "Usage: $0 <seeds> [ARGS...]"
    echo "       Schedules lots of runs"
    echo "       As with run_one.sh, seeds are used to populate a slurm array and ARGS"
    echo "        are passed directly to the executable"
}

seeds=$1
shift

base=$(realpath $(dirname $0)/../..)

export SILENT_SKIP_EXISTING=1

prefix(){
  printf "[%s] " "$(date)"
}

read -r -d '' bodies << EOM
  spider spider45 gecko babya ant salamander blokky park babyb garrix insect
  linkin longleg penguin pentapod queen squarish snake stingray tinlicker
  turtle ww zappa
EOM

(
  for body in $bodies
  do
    echo $body $seeds --body $body $@
  done
) | while read cmd
do
  while [ $(squeue -u kgd | wc -l) -gt 20 ]
  do
    prefix
    printf "Waiting for some room in queue\r"
    sleep 10
  done

  prefix
  $(dirname $0)/run_one.sh $cmd
  sleep 1
done