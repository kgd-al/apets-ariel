#!/bin/bash

user=robohat
host=${ROBOT:-kilo}
base=$user@$host:kgd

update(){
  dir=$1
  cd "../$dir"
  shift
  echo "Updating from $(pwd): $*"
  rsync -avzhP --prune-empty-dirs -f '- *.pyc' -f '- .*' "$@" "$base/$dir"
}

update apets-ariel src scripts pyproject.toml

line
update ariel src pyproject.toml setup.py
