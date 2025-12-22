#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} tensorflow-gpu'
target_dir=./dali/test/python

test_body() {
    # The package name can be nvidia-dali-tf-plugin,  nvidia-dali-tf-plugin-weekly or  nvidia-dali-tf-plugin-nightly
    pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true


    # No plugin installed, should fail
    ${python_new_invoke_test} test_dali_tf_plugin.TestDaliTfPluginLoadFail

    # Remove the old and installing "current" dali tf (built against installed TF)
    pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true

    pip install --upgrade ../../../nvidia_dali_tf_plugin*.tar.gz --no-build-isolation
    ${python_new_invoke_test} test_dali_tf_plugin.TestDaliTfPluginLoadOk

    # DALI TF run
    ${python_new_invoke_test} test_dali_tf_plugin_run
}

pushd ../..
source ./qa/test_template.sh
popd
