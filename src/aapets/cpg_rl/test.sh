#!/bin/bash

errors=0
items=0

run_one(){
  echo python -m aapets.cpg_rl.main --seed 0 \
    --trainer $1 --arch $2 --reward $3 \
    $4
    --overwrite --budget 10 --data-folder tmp/cpg_rl/$5 || true
  errors=$(($errors + $?))
  items=$(($item + 1))
}

(
  for neighborhood in 0 2 4 6
  do
    run_one cma cpg speed --cpg-neighborhood $neighborhood cpg-$i
  done

  for trainer in cma ppo
  do
    for width in 1 2 4 8 16 32 64 #128
    do
      run_one $trainer mlp speed --mlp-width $width --mlp-depth 0 $i mlp-0-$width
      for depth in 1 2
      do
        run_one $trainer mlp speed --mlp-width $width --mlp-depth $depth $i mlp-0-$width
      done
    done
  done
) > .tasks

print "$errors errors out of $items items"
