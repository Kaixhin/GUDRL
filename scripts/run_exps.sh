#!/bin/bash

MODE="${1:-online}"

for seed in {1..20}; do
  python main.py --seed $seed --mode $MODE
done
