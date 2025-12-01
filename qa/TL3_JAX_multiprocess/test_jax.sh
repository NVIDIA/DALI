#!/bin/bash -e

function CLEAN_AND_EXIT {
    exit $1
}

# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

install_cuda_compat

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0

python -c "import jax; print(jax.devices()); assert jax.device_count() > 0"

echo "Test one GPU per process"
CUDA_VISIBLE_DEVICES="1" timeout -k 60s 60s python jax_client.py --id 1 --size 8 &
CUDA_VISIBLE_DEVICES="2" timeout -k 60s 60s python jax_client.py --id 2 --size 8 &
CUDA_VISIBLE_DEVICES="3" timeout -k 60s 60s python jax_client.py --id 3 --size 8 &
CUDA_VISIBLE_DEVICES="4" timeout -k 60s 60s python jax_client.py --id 4 --size 8 &
CUDA_VISIBLE_DEVICES="5" timeout -k 60s 60s python jax_client.py --id 5 --size 8 &
CUDA_VISIBLE_DEVICES="6" timeout -k 60s 60s python jax_client.py --id 6 --size 8 &
CUDA_VISIBLE_DEVICES="7" timeout -k 60s 60s python jax_client.py --id 7 --size 8 &
CUDA_VISIBLE_DEVICES="0" timeout -k 60s 60s python jax_server.py --size 8 &

wait $(jobs -p)

echo "Test multiple GPUs per process"
CUDA_VISIBLE_DEVICES="4,5,6,7" timeout -k 60s 60s python jax_client.py --id 1 --size 2 &
CUDA_VISIBLE_DEVICES="0,1,2,3" timeout -k 60s 60s python jax_server.py --size 2 &

wait $(jobs -p)

CLEAN_AND_EXIT ${PIPESTATUS[0]}
