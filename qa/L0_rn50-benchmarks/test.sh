#!/bin/bash
set -e
# cd "$( dirname "${BASH_SOURCE[0]}" )"

cd /opt/ndll/build
./ndll/run_benchmarks --benchmark_filter="RN50*"
