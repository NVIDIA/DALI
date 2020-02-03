#!/bin/bash -e
# used pip packages
pip_packages="nose tensorflow-gpu"
target_dir=./dali/test/python

test_body() {
    # Manually removing the supported plugin so that it fails
    lib_dir=$(python -c 'import nvidia.dali.sysconfig as sc; print(sc.get_lib_dir())')
    rm -rf $lib_dir/plugin/*.so

    # No plugin installed, should fail
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadFail

    # Remove the old and installing "current" dali tf (built against installed TF)
    pip list | grep nvidia-dali-tf-plugin | xargs pip uninstall -y

    pip install --upgrade ../../../nvidia-dali-tf-plugin*.tar.gz
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # DALI TF run
    nosetests --verbose test_dali_tf_plugin_run.py
}

pushd ../..
source ./qa/test_template.sh
popd
