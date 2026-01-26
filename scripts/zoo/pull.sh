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
path=data/zoo
base=$user@$host:$path

summary_data=$path/summaries.csv
champions=$path/__champions/

#info=""
info=--info=progress2
log=.rsync.log

ssh $user@$host bash <<EOF
  rm -r $champions
  mkdir -p $champions

  for f in ~/$path/[^_]*/
  do
    awk -F, -vc=fitness 'NR == 1 {
      ci = -1
      for (i=1; i<=NF; i++) {
        if (\$i == c) {
          ci = i
          break
        }
      }
      if (ci == -1) {
        print "Could not find " c " in " \$0
        exit 1
      }
    } FNR == 2{
      print \$1 " " \$ci
    }' \$f/*/summary.csv | sort -k2,2g | tail -n 1 | cut -d ' ' -f 1
  done | while read f
  do
    body=\$(awk -F/ '{print \$(NF-1)}' <<< \$f)
    ln -s \$f/ $champions/\$body
  done
  ls -l $champions

  awk ' NR == 1 {
    print \$0
  } FNR == 2 {
    print \$0
  }' $path/*/*/summary.csv > $summary_data

EOF

(
  set -x;
  rsync -avzh -L $info $base/ remote/zoo --prune-empty-dirs --stats \
    -f '+ __champions/' -f '+ __champions/**' -f '+ summaries.csv' -f '- *' \
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
