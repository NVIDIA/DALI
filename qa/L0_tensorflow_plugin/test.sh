#!/bin/bash -e
# used pip packages
pip_packages="nose tensorflow-gpu"

pushd ../..

source qa/setup_dali_extra.sh
cd dali/test/python

test_body() {
    # Manually removing the supported plugin so that it fails
    lib_dir=$(python -c 'import nvidia.dali.sysconfig as sc; print(sc.get_lib_dir())')
    rm -rf $lib_dir/plugin/*.so

    # No plugin installed, should fail
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadFail

    # Installing "current" dali tf (built against installed TF)
    pip install ../../../nvidia-dali-tf-plugin*.tar.gz
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk
}

source ../../../qa/test_template.sh

popd
