#!/bin/bash -e
# used pip packages
# use TF that is installed from conda when DALI is installed
pip_packages='${python_test_runner_package} tensorflow-gpu'
target_dir=./dali/test/python

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

test_body() {
    ${python_new_invoke_test} test_dali_tf_plugin:TestDaliTfPluginLoadOk

    # DALI TF run
    ${python_new_invoke_test} test_dali_tf_plugin_run

    # DALI TF DATASET run
    ${python_new_invoke_test} test_dali_tf_dataset

    ${python_new_invoke_test} test_dali_tf_dataset_shape
}

pushd ../..
source ./qa/test_template.sh
popd
