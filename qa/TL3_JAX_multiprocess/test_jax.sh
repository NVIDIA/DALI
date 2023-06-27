#!/bin/bash -e

function CLEAN_AND_EXIT {
    exit $1
}

pushd /opt/dali/dali/test/python/

# Multiprocess tests
export NCCL_DEBUG=INFO

python -c "import jax; print(jax.devices()); assert jax.device_count() > 0"

CLEAN_AND_EXIT ${PIPESTATUS[0]}
