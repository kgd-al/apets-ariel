#!/bin/bash

if [ $# -eq 0 ]
then
  echo "No remote target folder specified"
  exit 1
fi

target=$1
shift 1

set -x
rsync -avzh --info=progress2 kgd@hex:~/data/symmetry/$target remote/symmetry \
  -f '+ */' -f '- _*/' -f '+ champion.*' -f '+ *.png' -f '+ novelty.pkl' \
  -f '- learning.csv' -f '+ *.csv' -f '+ slurm.*' \
  -f '- *' $@
