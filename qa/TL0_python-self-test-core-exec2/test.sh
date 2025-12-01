#!/bin/bash -e
export DALI_USE_EXEC2=1
pushd ../TL0_python-self-test-core
bash -e ./test_nofw.sh
bash -e ./test_pytorch.sh
popd
