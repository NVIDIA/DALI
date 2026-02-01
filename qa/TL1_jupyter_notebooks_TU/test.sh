#!/bin/bash -e

# used pip packages
pip_packages='jupyter matplotlib numpy'
target_dir=./docs/examples/

test_body() {
    test_files=(
        "sequence_processing/optical_flow/pipeline_mode.ipynb"
        "sequence_processing/optical_flow/dynamic_mode.ipynb"
    )

    # test code
    for f in ${test_files[@]}; do
        jupyter nbconvert --to notebook --inplace --execute \
                        --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                        --ExecutePreprocessor.timeout=600 $f
    done
}

pushd ../..
source ./qa/test_template.sh
popd
