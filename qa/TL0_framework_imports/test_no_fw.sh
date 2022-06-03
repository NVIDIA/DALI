#!/bin/bash -e
# used pip packages

pip_packages=''

test_body() {
    # test code
    export CUDA_VISIBLE_DEVICES=
    echo "---------Testing DALI----------"
    ( set -x && python -c "import nvidia.dali" )
    unset CUDA_VISIBLE_DEVICES
}

pushd ../../
source ./qa/test_template.sh
popd

