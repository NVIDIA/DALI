#!/bin/bash -e
export DALI_USE_NEW_THREAD_POOL=1
pushd ../TL0_python-self-test-core
bash -e ../TL0_python-self-test-core/test.sh
popd
