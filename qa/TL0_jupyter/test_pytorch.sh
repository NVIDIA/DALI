#!/bin/bash -e
# used pip packages
pip_packages="pillow jupyter numpy matplotlib torch torchvision"
target_dir=./docs/examples

test_body() {
    jupyter nbconvert --to notebook --inplace --execute \
                      --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                      --ExecutePreprocessor.timeout=300 custom_operations/python_operator.ipynb
}

pushd ../..
source ./qa/test_template.sh
popd
