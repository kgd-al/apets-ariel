#!/bin/bash

tasks=.tasks
results=$tasks.results.tsv
detailed_results=$tasks.results.csv

run_one(){
  trainer=$1
  arch=$2
  reward=$3
  name=$4
  shift 4
  echo python -m aapets.cpg_rl.main --seed 0 \
    --trainer $trainer --arch $arch --reward $reward \
    $@ \
    --overwrite --budget 200 --duration 1 --threads 1 --data-folder tmp/cpg_rl/$trainer/$reward/$name/run-0/
}

(
  for reward in speed gym kernels
  do
    for neighborhood in 0 2 4 6
    do
      run_one cma cpg $reward cpg-$neighborhood --cpg-neighborhood $neighborhood
    done

    for trainer in cma ppo
    do
      run_one $trainer mlp $reward mlp-0-0 --mlp-width 0 --mlp-depth 0 $i
      for width in 1 2 4 8 16 32 64 128
      do
        for depth in 1 2
        do
          run_one $trainer mlp $reward mlp-$depth-$width --mlp-width $width --mlp-depth $depth $i
        done
      done
    done
  done
) | grep -v -e "--trainer cma.*--mlp-width 64 --mlp-depth 2" -e "--trainer cma.*--mlp-width 128" > $tasks

parallel -j 4 --bar --eta --progress --joblog $results --results $detailed_results $@ < $tasks

items=$(tail -n +2 $results | wc -l)
errors=$(awk -F '\t' 'NR>1{errors += ($7!=0)}END{print errors}' $results)
echo "$errors errors out of $items items"

if [ $errors -ne 0 ]
then
  awk '$7!=0{print}' $results
fi

printf "\a"
