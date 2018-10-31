#!/bin/bash -e
# used pip packages
pip_packages="jupyter numpy matplotlib opencv-python"

pushd ../..

# attempt to run jupyter on all example notebooks
mkdir -p idx_files

cd docs/examples

test_body() {
    # test code
    ls *.ipynb | xargs -i jupyter nbconvert \
                   --to notebook --inplace --execute \
                   --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                   --ExecutePreprocessor.timeout=300 {}
}

source ../../qa/test_template.sh

popd
