#!/bin/bash -e
# used pip packages
pip_packages='jupyter numpy jax flax'
target_dir=./docs/examples

test_body() {
    test_files=(
        "frameworks/jax/jax-basic_example.ipynb",
        "custom_operations/jax_operator_basic.ipynb",
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
