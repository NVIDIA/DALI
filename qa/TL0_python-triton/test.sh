#!/bin/bash -e


pip_packages="nose"
target_dir=./dali/python/nvidia/dali/test


test_body() {
  for test in *_test.py ; do
    [[ -e "$test" ]] || break  # no *_test.py files
    nosetests -s -v "$test"
  done
}


pushd ../..
source ./qa/test_template.sh
popd
