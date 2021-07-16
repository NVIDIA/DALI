#!/bin/bash -e
# used pip packages
pip_packages="pillow jupyter numpy matplotlib torch torchvision webdataset"
target_dir=./docs/examples

test_body() {
    jupyter nbconvert --to notebook --inplace --execute \
                      --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                      --ExecutePreprocessor.timeout=300 custom_operations/python_operator.ipynb
    jupyter nbconvert --to notebook --inplace --execute \
                      --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                      --ExecutePreprocessor.timeout=300 use_cases/webdataset-externalsource.ipynb
}

pushd ../..
source ./qa/test_template.sh
popd
