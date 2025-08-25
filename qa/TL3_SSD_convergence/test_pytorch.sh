#!/bin/bash -e

set -o errexit
set -o pipefail

function CLEAN_AND_EXIT {
    ((IS_TMP_DIR)) && rm -rf ${DATA_DIR} || true
    exit $1
}

# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

install_cuda_compat

cd /opt/dali/docs/examples/use_cases/pytorch/single_stage_detector/

pip install mlperf_compliance Cython
pip install git+https://github.com/NVIDIA/cocoapi.git#subdirectory=PythonAPI

NUM_GPUS=$(nvidia-smi -L | wc -l)

export DATA_DIR=/data/coco/coco-2017/coco2017
export IS_TMP_DIR=0
if [ -f "/data/coco/coco-2017/coco2017/train2017.zip" ]; then
    apt update && apt install -y unzip
    export DATA_DIR=$(mktemp -d)
    export IS_TMP_DIR=1
    pushd ${DATA_DIR}
    cp /data/coco/coco-2017/coco2017/train2017.zip . &
    cp /data/coco/coco-2017/coco2017/val2017.zip . &
    cp /data/coco/coco-2017/coco2017/annotations_trainval2017.zip . &
    wait
    unzip -q train2017.zip &
    unzip -q val2017.zip &
    unzip -q annotations_trainval2017.zip &
    wait
    popd
fi

LOG=dali.log

SECONDS=0

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0

# Prevent OOM due to fragmentation on 16G machines
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:4096
torchrun --nproc_per_node=${NUM_GPUS} main.py --backbone resnet50 --warmup 300 --bs 64 --eval-batch-size 8 --data /coco --data ${DATA_DIR} --data_pipeline dali --target 0.25 2>&1 | tee $LOG

RET=${PIPESTATUS[0]}
echo "Training ran in $SECONDS seconds"
if [[ $RET -ne 0 ]]; then
    echo "Error in training script."
    CLEAN_AND_EXIT 2
fi

CLEAN_AND_EXIT 0
