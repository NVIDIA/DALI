#!/bin/bash -ex

source test_body.sh

pushd ../../dali/test/python

test_body

popd