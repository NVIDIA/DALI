#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} tensorflow-gpu'
target_dir=./dali/test/python

test_body() {
    # The package name can be nvidia-dali-tf-plugin,  nvidia-dali-tf-plugin-weekly or  nvidia-dali-tf-plugin-nightly
    pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true


    # No plugin installed, should fail
    ${python_invoke_test} test_dali_tf_plugin.py:TestDaliTfPluginLoadFail

    # Remove the old and installing "current" dali tf (built against installed TF)
    pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true

    pip install --upgrade ../../../nvidia_dali_tf_plugin*.tar.gz
    ${python_invoke_test} test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # DALI TF run
    ${python_invoke_test} test_dali_tf_plugin_run.py
}

pushd ../..
source ./qa/test_template.sh
popd
