#!/bin/bash -e
source ../TL1_tensorflow_dataset/test_impl.sh

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

pushd ../..
source ./qa/test_template.sh
popd
