#!/bin/bash -e

# Free-threaded Python is incompatible with numpy<2.
# Check if Python is compiled with --disable-gil and
# set NumPy version accordingly.
set +e
python3 -c 'import sysconfig ; exit(sysconfig.get_config_var("Py_GIL_DISABLED"))'
if [ $? -ne 0  ]; then
    pip_packages='${python_test_runner_package} numpy>=2' ;
else
    pip_packages='${python_test_runner_package} numpy' ;
fi
set -e

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
