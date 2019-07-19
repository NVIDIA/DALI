#!/bin/bash -e
# used pip packages
pip_packages="nose"

pushd ../..

source qa/setup_dali_extra.sh
cd dali/test/python

USE_CUDA_VERSION=$(cat /usr/local/cuda/version.txt | sed 's/.*Version \([0-9]\+\)\.\([0-9]\+\).*/\1/')
test "$USE_CUDA_VERSION" = "10" && export TENSORFLOW_VERSIONS="1.13.1 1.14"
test "$USE_CUDA_VERSION" = "9" && export TENSORFLOW_VERSIONS="1.7 1.8 1.9 1.10 1.11 1.12"

test_body() {
    # Manually removing the supported plugin so that it fails
    lib_dir=$(python -c 'import nvidia.dali.sysconfig as sc; print(sc.get_lib_dir())')
    rm -rf $lib_dir/plugin/*.so

    for tensorflow_ver in ${TENSORFLOW_VERSIONS}; do
        echo "Testing tensorflow-gpu==$tensorflow_ver"

        # No plugin installed, should fail
        nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadFail

        pip install "tensorflow-gpu==$tensorflow_ver"

        # Installing "current" dali tf (built against installed TF)
        pip install ../../../nvidia-dali-tf-plugin*.tar.gz
        nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

        # DALI TF run
        nosetests --verbose test_dali_tf_plugin_run.py

        pip uninstall -y "tensorflow-gpu"
    done
}

source ../../../qa/test_template.sh

popd
