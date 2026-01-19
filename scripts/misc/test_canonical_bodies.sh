#!/bin/bash

activate(){
  source ${PYENV_ROOT}/versions/$1/bin/activate
}

clear

cd ../apets
activate venv-revolve
python src/apets/hack/cma_es/evolve.py --rerun -o remote/spider45/cma/kernels/cpg-0/run-0 --reward distance --arch cpg --neighborhood 0 -T 3600 --start-paused --body $1 &

cd ../apets-ariel
activate venv-ariel
python -m aapets.bin.rerun --no-check-performance --verbosity 0 --viewer PASSIVE --no-auto-start --default-body $1

wait

clear

f=src/aapets/common/canonical_bodies.py
echo "Done: $(grep 'def body_.*()' $f | wc -l), $(grep '_v1()' $f | wc -l) remaining"
