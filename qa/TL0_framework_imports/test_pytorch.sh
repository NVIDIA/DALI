#!/bin/bash -e
# used pip packages

pip_packages='pillow torchvision torch'

test_body() {
    # test code
    echo "---------Testing PYTORCH;DALI----------"
    ( set -x && python -c "import torch; import nvidia.dali.plugin.pytorch" )
    echo "---------Testing DALI;PYTORCH----------"
    ( set -x && python -c "import nvidia.dali.plugin.pytorch; import torch" )
}

pushd ../../
source ./qa/test_template.sh
popd
