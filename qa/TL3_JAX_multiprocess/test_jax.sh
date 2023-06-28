#!/bin/bash -e

function CLEAN_AND_EXIT {
    exit $1
}

python -c "import jax; print(jax.devices()); assert jax.device_count() > 0"

CLEAN_AND_EXIT ${PIPESTATUS[0]}
