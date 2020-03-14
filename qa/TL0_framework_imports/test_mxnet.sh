#!/bin/bash -e
# used pip packages

pip_packages="mxnet"

test_body() {
    # test code
    echo "---------Testing MXNET;DALI----------"
    ( set -x && python -c "import mxnet; import nvidia.dali.plugin.mxnet" )
    echo "---------Testing DALI;MXNET----------"
    ( set -x && python -c "import nvidia.dali.plugin.mxnet; import mxnet" )
}

pushd ../../
source ./qa/test_template.sh
popd
