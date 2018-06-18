#!/bin/bash -e

pushd ../..

pip install numpy==1.11.1

cd dali/benchmark
python resnet50_bench.py

popd
