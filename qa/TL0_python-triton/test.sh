#!/bin/bash -e
target_dir=./dali/python/test

test_body() {
    nosetests -s -v "$(ls ./*_test.py)"
}

pushd ../..
source ./qa/test_template.sh
popd
