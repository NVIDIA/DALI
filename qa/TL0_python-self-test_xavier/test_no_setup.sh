#!/bin/bash -ex

if [ -n "$gather_pip_packages" ]
then
    # early exit
    return 0
fi

source test_body.sh

pushd ../../dali/test/python

run_all

popd
