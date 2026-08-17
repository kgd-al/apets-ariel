#!/bin/bash

lines=${LINES:-10}

# echo "Running"
# ls -t ~/data/zoo/_slurm_logs/*.out | head -n$lines | while read f
# do
#     printf "%s\t%s\n" \
#         "$(sed 's|.*/cma-zoo-\(.*\).out|\1|' <<< $f)" \
#         "$(tail -n1 $f)"
# done | nl

# echo
echo "Completed"
ls -t ~/data/zoo/[a-z]*/*/slurm.out | xargs grep "Completed" \
    | cut -d ':' -f 1 | cut -d/ -f 6 | sort | uniq -c
