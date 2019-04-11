#!/bin/bash -e

source ../setup_test.sh

# used pip packages
pip_packages="jupyter"

pushd ../..

# attempt to run jupyter on all example notebooks
mkdir -p idx_files

cd docs/examples

test_body() {
    test_files="optical_flow_example.ipynb"

    # test code
    echo $test_files | xargs -i jupyter nbconvert \
                   --to notebook --inplace --execute \
                   --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                   --ExecutePreprocessor.timeout=600 {}
}

source ../../qa/test_template.sh

popd
