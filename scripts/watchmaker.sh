#!/bin/bash

python -m aapets.watchmaker.bin.evo --data-folder results/watchmaker \
  --max-evaluations 201 --population-size 5 --layout 0xA5 \
  --body spider45 --duration 30 --speed-up 6 --camera-angle 45 \
  --mutation-range 2 --mutation-scale .5 --no-symlink --plot-extension png \
  $@
