#!/usr/bin/env zsh

for fp in ./data/*.json; do
  fname=${fp:t:r}
  mongoimport --db sales_ai --collection ${fname} --file $fp
done
