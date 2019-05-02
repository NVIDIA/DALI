#!/bin/bash -e
# used pip packages
pip_packages="nose tensorflow-gpu"

pushd ../..

source qa/setup_test.sh
cd dali/test/python

test_body() {
    # Load plugin OK (supported version of TF in the prebuilt plugins)
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # Manually removing the supported plugin so that it fails
    lib_dir=$(python -c 'import nvidia.dali.sysconfig as sc; print(sc.get_lib_dir())')
    pushd $lib_dir/plugin
    mkdir tmp
    mv *.so tmp
    popd
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadFail

    # Installing "current" dali tf (built against installed TF)
    pip install ../../../nvidia-dali-tf-plugin*.tar.gz
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # Restore plugins (still works)
    mv $lib_dir/plugin/tmp/* $lib_dir/plugin
    rm -rf $lib_dir/plugin/tmp
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk
}

source ../../../qa/test_template.sh

popd
