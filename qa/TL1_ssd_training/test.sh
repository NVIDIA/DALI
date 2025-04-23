#!/bin/bash -e
# used pip packages
pip_packages='numpy pillow torch torchvision mlperf_compliance matplotlib Cython pycocotools'
target_dir=./docs/examples/use_cases/pytorch/single_stage_detector/

test_body() {
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
    torchrun --nproc_per_node=${NUM_GPUS} main.py --backbone resnet50 --warmup 300 --bs 256 --eval-batch-size 8 --epochs 4 --data ${DATA_DIR} --data_pipeline dali --target 0.085
    ((IS_TMP_DIR)) && rm -rf ${DATA_DIR} || true
}

pushd ../..
source ./qa/test_template.sh
popd
