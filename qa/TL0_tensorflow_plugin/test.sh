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
    # The package name can be nvidia_dali_tf_plugin,  nvidia_dali_tf_plugin-weekly or  nvidia_dali_tf_plugin-nightly
    pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true

    # No plugin installed, should fail
    ${python_invoke_test} test_dali_tf_plugin.py:TestDaliTfPluginLoadFail

    # Installing "current" dali tf (built against installed TF)
    pip install ../../../nvidia_dali_tf_plugin*.tar.gz
    ${python_invoke_test} test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # Installing "current" dali tf (built against installed TF) - force rebuild without DALI using internal stubs
    # and then install DALI again
    pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true
    pip uninstall -y `pip list | grep nvidia-dali | cut -d " " -f1` || true
    DALI_TF_ALWAYS_BUILD=1 pip install --no-deps ../../../nvidia_dali_tf_plugin*.tar.gz
    pip install ../../../nvidia_dali_*.whl
    ${python_invoke_test} test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # DALI TF run
    ${python_invoke_test} test_dali_tf_plugin_run.py

    # DALI TF DATASET run
    ${python_invoke_test} test_dali_tf_dataset.py
    ${python_invoke_test} test_dali_tf_conditionals.py
    ${python_new_invoke_test} checkpointing.test_dali_checkpointing_tf_plugin
    if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
        ${python_invoke_test} test_dali_tf_dataset_shape.py
        ${python_invoke_test} test_dali_tf_dataset_eager.py
        ${python_invoke_test} test_dali_tf_dataset_graph.py
    fi

    # DALI TF + dynamic executor
    ${python_invoke_test} test_dali_tf_exec2.py
}

pushd ../..
source ./qa/test_template.sh
popd
