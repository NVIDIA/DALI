#!/bin/bash -e

pushd ../TL0_python-self-test-core
bash -e ./test.sh
popd

pushd ../TL0_python-self-test-readers-decoders
bash -e ./test.sh
popd

pushd ../TL0_python-self-test-operators
bash -e ./test.sh
popd
