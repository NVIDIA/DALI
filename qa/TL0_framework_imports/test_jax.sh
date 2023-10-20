#!/bin/bash -e
# used pip packages

pip_packages='jax'

test_body() {
    # test code
    echo "---------Testing JAX;DALI----------"
    ( set -x && python -c "import jax; import nvidia.dali.plugin.jax" )
    echo "---------Testing DALI;JAX----------"
    ( set -x && python -c "import nvidia.dali.plugin.jax; import jax" )
}

pushd ../../
source ./qa/test_template.sh
popd
