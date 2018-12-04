#!/bin/bash
# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed> <target threshold>

SEED=${1:-1}
TARGET=${2:-0.212}

time stdbuf -o 0 \
  python3 train.py --seed $SEED --threshold $TARGET | tee run.log.$SEED
