#!/bin/bash

set -euo pipefail
export PYTHONPATH=.

if [ -z ${VIRTUAL_ENV+x} ]
then
  echo "Not in a virtual environment. Aborting"
  exit 1
fi

user=kgd
host=hex
base=$user@$host:data/zoo

#info=""
info=--info=progress2
log=.rsync.log
(
  set -x;
  rsync -avzh $info $base remote --prune-empty-dirs --stats
) | tee $log

transferred=$(grep "Total transferred file size" $log | cut -d ' ' -f2)
if [ transferred != "0" ]
then
  generate="--regenerate"
else
  generate=""
fi

python -m aapets.zoo.analyze $generate

[ $# -gt 0 ] && [ $1 == "--vlc" ] && vlc --no-random --no-loop remote/zoo/_best/*.mp4 2> /dev/null
