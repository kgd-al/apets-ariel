#!/bin/bash

usage(){
    echo "Usage: $0 <folder> [fields]"
    echo 
    echo "       Monitor completion rates for experiments under <folder>"
    echo "       Monitor completion rates for experiments under <folder>"
}

if [ $# -lt 1 ] || [ $# -gt 2 ]
then
    usage
    exit 1
fi

folder=$1
folders=$(find $folder -name "run-*")

fields=${2:-7-8}

printf "Running:\n"
for f in $folders
do
    [ ! -f $f/slurm.out ] && echo $f
done | cut -d/ -f $fields | tr / " " | sort | uniq -c

printf "\n\nCompleted:\n"

find $folder -name champion.zip | cut -d/ -f $fields | tr / " " | sort | uniq -c

aborted=$(for f in $folders
do 
    [ -f $f/slurm.out ] && [ ! -f $f/champion.zip ] && echo $f
done)

if [ -n "$aborted" ]
then
    printf "\n\nAborted:\n"
    printf ">>\e[31m%s\e[0m\n" $aborted
else
    printf "\n\e[32mNo failed runs \e[0;90m(yet?)\e[0m\n"
fi

