#!/bin/bash

target=$1
shift 1

set -x
rsync -avzh --info=progress2 kgd@hex:~/data/$target remote -f '+ */' -f '- _*/' -f '+ champion.*' -f '+ summary.csv' -f '+ progress.csv' -f '+ xrecentbest.dat' -f '- *' $@
