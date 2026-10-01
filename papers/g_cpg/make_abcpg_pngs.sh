#!/bin/sh
# Renders full.png + one transparent PNG per layer.
cd $(dirname $0)
for l in all cpg abcpg policy base loop; do
  pdflatex -interaction=nonstopmode -jobname="$l" "\def\layer{$l}\input{abcpg}" >/dev/null
  pdftocairo -png -transp -singlefile -r 400 "$l.pdf" "$l"
  rm $l.aux $l.log $l.pdf
done
