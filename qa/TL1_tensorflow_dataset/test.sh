#!/bin/bash -e
source test_impl.sh

# populate epilog and prolog with variants to enable/disable virtual env
# every test will be executed for bellow configs
prolog=(: enable_virtualenv)
epilog=(: disable_virtualenv)

pushd ../..
source ./qa/test_template.sh
popd
