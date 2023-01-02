#!/bin/bash -e

pushd ../TL0_python-self-test-core
bash -e ./test.sh
popd

pushd ../TL0_python-self-test-readers-decoders
bash -e ./test.sh
popd

pushd ../TL0_python-self-test-operators_1
bash -e ./test.sh
popd

pushd ../TL0_python-self-test-operators_2
bash -e ./test.sh
popd
