#!/bin/bash

if [ $# -eq 0 ]
then
  echo "No remote target folder specified"
  exit 1
fi

target=$1
shift 1

set -x
rsync -avzh --info=progress2 kgd@hex:~/data/$target remote -f '+ */' -f '- _*/' -f '+ champion.*' -f '+ summary.csv' -f '+ progress.csv' -f '+ xrecentbest.dat' -f '- *' $@
