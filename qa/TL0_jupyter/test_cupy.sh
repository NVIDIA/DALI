#!/bin/bash -e
# used pip packages
pip_packages="jupyter numpy matplotlib cupy-cuda{cuda_v}"
target_dir=./docs/examples

test_body() {
    jupyter nbconvert --to notebook --inplace --execute \
                      --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                      --ExecutePreprocessor.timeout=300 custom_operations/gpu_python_operator.ipynb
}

pushd ../..
source ./qa/test_template.sh
popd
