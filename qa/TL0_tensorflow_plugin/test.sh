#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} tensorflow-gpu'
target_dir=./dali/test/python

# reduce the lenght of the sanitizers tests as much as possible
# use only one TF verion, don't test virtual env
if [ -n "$DALI_ENABLE_SANITIZERS" ]; then
    one_config_only=true
else
    # populate epilog and prolog with variants to enable/disable conda and virtual env
    # every test will be executed for bellow configs
    prolog=(: enable_virtualenv)
    epilog=(: disable_virtualenv)
fi

test_body() {
    # The package name can be nvidia-dali-tf-plugin,  nvidia-dali-tf-plugin-weekly or  nvidia-dali-tf-plugin-nightly
    pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true

    # No plugin installed, should fail
    ${python_invoke_test} test_dali_tf_plugin.py:TestDaliTfPluginLoadFail

    # Installing "current" dali tf (built against installed TF)
    pip install ../../../nvidia-dali-tf-plugin*.tar.gz
    ${python_invoke_test} test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # DALI TF run
    ${python_invoke_test} test_dali_tf_plugin_run.py

    # DALI TF DATASET run
    ${python_invoke_test} test_dali_tf_dataset.py
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        ${python_invoke_test} test_dali_tf_dataset_shape.py
        ${python_invoke_test} test_dali_tf_dataset_eager.py
        ${python_invoke_test} test_dali_tf_dataset_graph.py
    fi
}

pushd ../..
source ./qa/test_template.sh
popd
