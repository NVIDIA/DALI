#!/bin/bash -e
# used pip packages
pip_packages='pillow jupyter numpy matplotlib torch torchvision webdataset pyyaml'
target_dir=./docs/examples

test_body() {
    test_files=(
        "custom_operations/python_operator.ipynb"
        "use_cases/webdataset-externalsource.ipynb"
    )
    for f in ${test_files[@]}; do
        if [ -n "${DALI_ENABLE_SANITIZERS:-}" ]; then
            bash "$topdir/qa/TL0_jupyter/run_notebook_sanitizer.sh" \
                "$topdir" "${PYVER:0:1}" "$f"
        else
            jupyter nbconvert --to notebook --inplace --execute \
                            --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                            --ExecutePreprocessor.timeout=300 $f;
        fi
    done
}

pushd ../..
source ./qa/test_template.sh
popd
