#!/bin/bash -e

pip_packages="nose"
target_dir=./dali/python/nvidia/dali/test

test_body() {
    nosetests -s -v "$(ls ./*_test.py)"
}

pushd ../..
source ./qa/test_template.sh
popd
