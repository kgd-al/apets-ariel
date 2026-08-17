#!/bin/bash

squeue -h -u ${USER:-kgd} -o %F | sort | uniq | xargs scancel
