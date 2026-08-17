#!/bin/bash

lines=${LINES:-20}

for f in $(ls ~/data/g_cpg/[a-z]*/[a-z]*/*/*/learning.csv 2>/dev/null)
do
	[ ! -f $(dirname $f)/slurm.out ] && wc -l $f | cut -d/ -f 1,6-9 | tr / ' '
done \
	| awk '{$1=($1/10000)"%";print}' \
     	| column -t -R 0 \
       	| sort -k1,1g \
	| nl | sort -k1,1gr | head -n $lines;
       
echo
grep -rn Completed ~/data/g_cpg/*/*/*/*/slurm.out \
	| sed -e 's|/home/kgd/data/g_cpg/||' -e 's|/slurm.out.*:|\||' -e 's/with/|with/' \
	| tr '/' '|' | sort -r | nl | sort -r | column -t -s '|' | head -n $lines;

