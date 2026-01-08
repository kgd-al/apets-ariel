#!/bin/bash

activate(){
  source ${PYENV_ROOT}/versions/$1/bin/activate
}

cd ../apets
activate venv-revolve
python src/apets/hack/cma_es/evolve.py --rerun -o remote/spider45/cma/kernels/cpg-0/run-0 --reward distance --arch cpg --neighborhood 0 -T 3600 --body $1 &

cd ../apets-ariel
activate venv-ariel
python -m aapets.bin.rerun --check-performance False --verbosity 0 --viewer INTERACTIVE --default-body $1

wait
