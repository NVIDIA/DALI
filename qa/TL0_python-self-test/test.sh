#!/bin/bash -e

pushd ../TL0_python-self-test-core
./test.sh
popd

pushd ../TL0_python-self-test-operators
./test.sh
popd