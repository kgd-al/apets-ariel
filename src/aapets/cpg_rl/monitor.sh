#!/bin/bash

exp=${1:-cpg_rl}

W=$(tput cols)

#sprio #-u $others
#sshare -l

root=~/data
#root=~/itch

completed=$(ls $root/$exp/*/*/*/*/*/summary.csv 2>/dev/null | wc -l)
folders=$(ls -d $root/$exp/*/*/*/*/run* 2>/dev/null | wc -l)
echo "  Running" $(($folders - $completed))
echo "Completed" $completed
echo "    Total" $folders
echo

Nj=$(squeue -o %j | wc -L)
Ni=$(squeue -o %i | wc -L)
squeue -o "%.${Ni}i %.9P %.${Nj}j %.6u %.8T %.10M %R" | cut -c -$W

echo
for f in $root/$exp/*/*/*/*/*/slurm.err
do
	echo $(grep -v -e '^$' -e "libEGL" $f 2> /dev/null | wc -l) $f
done | grep -v '^0' | sort -k1,1gr
 


