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

echo "test.sh Sanity check: PYTHON_GIL=${PYTHON_GIL}"

bash -e ./test_nofw.sh
