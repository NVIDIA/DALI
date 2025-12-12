#!/bin/bash -e

echo "test.sh Sanity check: PYTHON_GIL=${PYTHON_GIL}"

bash -e ./test_nofw.sh
