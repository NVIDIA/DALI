#!/bin/bash -e
# used pip packages
pip_packages="nose tensorflow-gpu"
target_dir=./dali/test/python

test_body() {
    # The package name can be nvidia-dali-tf-plugin,  nvidia-dali-tf-plugin-weekly or  nvidia-dali-tf-plugin-nightly
    pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true


    # No plugin installed, should fail
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadFail

    # Remove the old and installing "current" dali tf (built against installed TF)
    pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true

    pip install --upgrade ../../../nvidia-dali-tf-plugin*.tar.gz
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # DALI TF run
    nosetests --verbose test_dali_tf_plugin_run.py
}

pushd ../..
source ./qa/test_template.sh
popd
