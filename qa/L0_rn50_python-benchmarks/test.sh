#!/bin/bash -e

pip install numpy==1.11.1

cd /opt/dali/ndll/benchmark
python resnet50_bench.py
