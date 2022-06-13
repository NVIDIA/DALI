#!/bin/bash -e
# used pip packages
pip_packages='jupyter numpy matplotlib cupy imageio'
target_dir=./docs/examples

test_body() {
    test_files=(
        "custom_operations/gpu_python_operator.ipynb"
        "general/data_loading/external_input.ipynb"
    )
    for f in ${test_files[@]}; do
        jupyter nbconvert --to notebook --inplace --execute \
                        --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                        --ExecutePreprocessor.timeout=300 $f;
    done
}

pushd ../..
source ./qa/test_template.sh
popd
