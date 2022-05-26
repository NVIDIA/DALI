#!/bin/bash -e
# used pip packages
pip_packages='jupyter numpy matplotlib cupy imageio'
target_dir=./docs/examples

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

test_body() {
    test_files=(
        "custom_operations/gpu_python_operator.ipynb"
        "general/data_loading/external_input.ipynb"
    )
    # workarround for cupy using the wrong version of libnvrtc-builtins.so
    # while cupy link to libnvrtc.so corresponding to the version it was built for
    # libnvrtc.so loads the libnvrtc-builtins.so based on the $PATH and for conda
    # it ends up using libnvrtc-builtins.so installed together with TensorFlow
    # which is the latest and there is an obvious mismatch between libnvrtc-builtins.so
    # and libnvrtc.so
    LIB_PATH=$(dirname $(which python))/../lib
    for f in $(ls $LIB_PATH/libnvrtc-builtins.so*); do
        mv $f $f.bak
    done
    for f in ${test_files[@]}; do
        jupyter nbconvert --to notebook --inplace --execute \
                        --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                        --ExecutePreprocessor.timeout=300 $f;
    done
    for f in $(ls $LIB_PATH/libnvrtc-builtins.so*); do
        mv $f ${f/.bak/}
    done
}

pushd ../..
source ./qa/test_template.sh
popd
