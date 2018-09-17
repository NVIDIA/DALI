#!/bin/bash -e
# used pip packages
pip_packages="numpy"

pushd ../..

cd dali/benchmark

test_body() {
    # test code
    python resnet50_bench.py
}

source ../../qa/test_template.sh

popd
