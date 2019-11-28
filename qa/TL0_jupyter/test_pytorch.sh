#!/bin/bash -e
# used pip packages
pip_packages="jupyter numpy matplotlib torch"
target_dir=./docs/examples

test_body() {
    # test code
    # dummy
    black_list_files="#"

    find python_operator/* -name "*.ipynb" | sed "/${black_list_files}/d" | xargs -i jupyter nbconvert \
                    --to notebook --inplace --execute \
                    --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                    --ExecutePreprocessor.timeout=300 {}
}

pushd ../..
source ./qa/test_template.sh
popd
