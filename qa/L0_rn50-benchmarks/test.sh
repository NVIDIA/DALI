#!/bin/bash -e

cd /opt/dali/build-*$PYV*
./ndll/run_benchmarks --benchmark_filter="RN50*"
