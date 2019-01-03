#!/bin/bash -e
# used pip packages
pip_packages="jupyter matplotlib opencv-python mxnet-cu90 tensorflow-gpu torchvision torch"

pushd ../..

# attempt to run jupyter on all example notebooks
mkdir -p idx_files

# We need cmake to run the custom plugin notebook
apt-get update
apt-get install -y --no-install-recommends cmake

cd docs/examples

test_body() {
    # test code
    find */* -name "*.ipynb" | xargs -i jupyter nbconvert \
                   --to notebook --inplace --execute \
                   --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                   --ExecutePreprocessor.timeout=600 {}
    find */* -name "main.py" | xargs -i python${PYVER:0:1} {} -t
}

source ../../qa/test_template.sh

popd
