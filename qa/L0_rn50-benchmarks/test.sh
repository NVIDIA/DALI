#!/bin/bash
set -e

cd /opt/ndll/build*$PYV*
./ndll/run_benchmarks --benchmark_filter="RN50*"
