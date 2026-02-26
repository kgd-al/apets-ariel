#!/bin/bash

run_one(){
  trainer=$1
  arch=$2
  reward=$3
  folder=$4
  shift 4
  echo python -m aapets.cpg_rl.main --seed 0 \
    --trainer $trainer --arch $arch --reward $reward \
    $@ \
    --overwrite --budget 1000 --data-folder tmp/cpg_rl/$folder
}

(
  for reward in speed lazy kernels
  do
    for neighborhood in 0 2 4 6
    do
      run_one cma cpg speed cpg-$neighborhood --cpg-neighborhood $neighborhood
    done

    for trainer in cma ppo
    do
      for width in 1 2 4 8 16 32 64 128
      do
        run_one $trainer mlp $reward mlp-0-$width --mlp-width $width --mlp-depth 0 $i
        for depth in 1 2
        do
          run_one $trainer mlp $reward mlp-0-$width --mlp-width $width --mlp-depth $depth $i
        done
      done
    done
  done
) | grep -v -e "--trainer cma.*--mlp-width 64 --mlp-depth 2" -e "--trainer cma.*--mlp-width 128" > .tasks

cat .tasks

errors=0
items=0
while read task
do
  (
    set -x
    echo $task
  )

  errors=$(($errors + $?))
  items=$(($items + 1))
done < .tasks

echo "$errors errors out of $items items"
