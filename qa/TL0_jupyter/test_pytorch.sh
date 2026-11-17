#!/bin/bash -e
# used pip packages
pip_packages='pillow jupyter numpy matplotlib torch torchvision webdataset pyyaml'
target_dir=./docs/examples

test_body() {
    kernel_args=()
    if [ -n "${DALI_ENABLE_SANITIZERS:-}" ]; then
        kernel_args+=('--ExecutePreprocessor.extra_arguments=["--IPKernelApp.capture_fd_output=False"]')
    fi

    test_files=(
        "custom_operations/python_operator.ipynb"
        "use_cases/webdataset-externalsource.ipynb"
    )
    for f in ${test_files[@]}; do
        jupyter nbconvert --to notebook --inplace --execute \
                        --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                        --ExecutePreprocessor.timeout=300 "${kernel_args[@]}" $f;
    done
}

pushd ../..
source ./qa/test_template.sh
popd
