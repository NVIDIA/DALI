#!/bin/bash -e

# PYTHON_GIL can be set to 0 only if Python is compiled with --disable-gil.
# Check if Python is compiled with --disable-gil.
set +e
python3 -c 'import sysconfig ; exit(sysconfig.get_config_var("Py_GIL_DISABLED"))'
# Set PYTHON_GIL accordingly.
if [ $? -ne 0  ]; then
        export PYTHON_GIL=0
fi
set -e

test_py_with_framework() {
    ${python_new_invoke_test} -A '!slow' -s free-threading
}

test_no_fw() {
    test_py_with_framework
}

run_all() {
  test_no_fw
}
