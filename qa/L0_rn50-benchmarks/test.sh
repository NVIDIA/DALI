#!/bin/bash -e

pushd ../..

cd build-*$PYV*
./ndll/run_benchmarks --benchmark_filter="RN50*"

popd
