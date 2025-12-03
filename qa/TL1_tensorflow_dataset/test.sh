#!/bin/bash -e

# Tensorflow tests are incompatible with Python 3.13.
# Check Python version and run test accordingly.
set +e
python -c '
import sys
if sys.version_info == (3, 13):
    sys.exit(1)
'
if [ $? -ne 0 ]; then
    exit 0
fi
set -e

source test_impl.sh

# populate epilog and prolog with variants to enable/disable virtual env
# every test will be executed for bellow configs
prolog=(: enable_virtualenv)
epilog=(: disable_virtualenv)

pushd ../..
source ./qa/test_template.sh
popd
