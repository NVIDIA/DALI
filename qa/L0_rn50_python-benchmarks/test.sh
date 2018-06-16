#!/bin/bash -e

pushd ../..

pip install numpy==1.11.1

cd ndll/benchmark
python resnet50_bench.py

popd
