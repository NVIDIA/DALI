#!/bin/bash -e

source ../setup_dali_extra.sh

# used pip packages
pip_packages="jupyter matplotlib numpy"

pushd ../../docs/examples

test_body() {
    test_files=("optical_flow/optical_flow_example.ipynb")

    # test code
    echo $test_files | xargs -i jupyter nbconvert \
                   --to notebook --inplace --execute \
                   --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                   --ExecutePreprocessor.timeout=600 {}
}

source ../../qa/test_template.sh

popd
