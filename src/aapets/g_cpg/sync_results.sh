#!/bin/bash

remote_base="kgd@hex:~/data/g_cpg/"
local_base="remote/g_cpg"

if [ $# -eq 0 ]
then
  echo "No remote target folder specified"
  echo "Usage: $0 <folder>"
  echo "       will copy data from $remote_base<folder> in $local_base"
  exit 1
fi

target=$1
shift 1

set -x
rsync -avzh --info=progress2 $remote_base/$target $local_base \
  -f '+ */' -f '- _*/' -f '+ champion*' -f '+ *.png' -f '+ novelty.pkl' \
  -f '- learning.csv' -f '+ *.csv' -f '+ slurm.*' \
  -f '- *' $@
