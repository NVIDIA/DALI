#!/bin/bash -e

function CLEAN_AND_EXIT {
    exit $1
}

pushd /opt/dali/test/python/

# Multiprocess tests
export NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES="1" python jax/jax_client.py &
CUDA_VISIBLE_DEVICES="0" python jax/jax_server.py

CLEAN_AND_EXIT ${PIPESTATUS[0]}
