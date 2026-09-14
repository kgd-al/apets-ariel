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
folders=

fields=${2:-7-8}
format(){
    cut -d/ -f $fields | tr / " " | sort | uniq -c
}

declare -a finished=()
declare -a running=()
declare -a failed=()

for f in $(find $folder -name "run-*")
do
    if [ -f $f/slurm.out ]
    then
        if [ -f $f/champion.zip ]
        then
            finished+=($f)
        else
            failed+=($f)
        fi
    else
        running+=($f)
    fi
done

if (( ${#running[@]} > 0))
then
    printf "Running:\n"
    printf "%s\n" "${running[@]}" | format
    printf "\n\n"
else
    printf "No jobs running\n"
fi

if (( ${#finished[@]} > 0))
then
    printf "Completed:\n"
    printf "%s\n" "${finished[@]}" | format
    printf "\n\n"
else
    printf "No jobs completed\n"
fi

if (( ${#failed[@]} > 0))
then
    printf "Failed:\n"
    printf "%s\n" "${failed[@]}" | sed 's/^.*$/\e[31m&\e[0m'
    printf "\n\n"
else
    printf "No jobs failed\n"
fi



