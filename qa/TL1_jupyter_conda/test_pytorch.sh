#!/bin/bash -e
# used pip packages
pip_packages='pillow jupyter numpy matplotlib torch torchvision webdataset pyyaml'
target_dir=./docs/examples

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

test_body() {
    test_files=(
        "custom_operations/python_operator.ipynb"
        "use_cases/webdataset-externalsource.ipynb"
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
