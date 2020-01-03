#!/bin/bash -e
# used pip packages

# TODO(janton): remove explicit pillow version installation when torch fixes the issue with PILLOW_VERSION not being defined
pip_packages="pillow==6.2.2 torchvision torch"

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
