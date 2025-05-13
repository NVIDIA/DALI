#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} numpy'
target_dir=./dali/test/python

# don't test conda with snitizers
if [ -z "$DALI_ENABLE_SANITIZERS" ]; then
  # populate epilog and prolog with variants to enable/disable conda
  # every test will be executed for bellow configs
  prolog=(: enable_conda)
  epilog=(: disable_conda)
fi

test_body() {
    ${python_invoke_test} test_plugin_manager.py
}

pushd ../..
source ./qa/test_template.sh
popd
